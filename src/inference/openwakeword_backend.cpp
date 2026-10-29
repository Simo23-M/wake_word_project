/**
 * @file openwakeword_backend.cpp
 * @brief Implementation of OpenWakeWord inference pipeline.
 */

#include "openwakeword_backend.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <stdexcept>

namespace wake_word {

/**
 * @brief ONNX Runtime session management for three models.
 */
struct OpenWakeWordBackend::Impl {
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::MemoryInfo> memory_info;

    // Melspectrogram model
    std::unique_ptr<Ort::Session> melspec_session;
    std::vector<const char*> melspec_input_names;
    std::vector<const char*> melspec_output_names;

    // Embedding model
    std::unique_ptr<Ort::Session> embedding_session;
    std::vector<const char*> embedding_input_names;
    std::vector<const char*> embedding_output_names;

    // Wake word model
    std::unique_ptr<Ort::Session> wakeword_session;
    std::vector<const char*> wakeword_input_names;
    std::vector<const char*> wakeword_output_names;

    // Stored names (ONNX Runtime requires stable pointers)
    std::vector<std::string> melspec_input_names_storage;
    std::vector<std::string> melspec_output_names_storage;
    std::vector<std::string> embedding_input_names_storage;
    std::vector<std::string> embedding_output_names_storage;
    std::vector<std::string> wakeword_input_names_storage;
    std::vector<std::string> wakeword_output_names_storage;
};

OpenWakeWordBackend::OpenWakeWordBackend(const OpenWakeWordConfig& config)
    : impl_(std::make_unique<Impl>())
    , config_(config)
{
    impl_->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "openwakeword");
    impl_->memory_info = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
    );
}

OpenWakeWordBackend::~OpenWakeWordBackend() = default;

OpenWakeWordBackend::OpenWakeWordBackend(OpenWakeWordBackend&&) noexcept = default;
OpenWakeWordBackend& OpenWakeWordBackend::operator=(OpenWakeWordBackend&&) noexcept = default;

namespace {
    /**
     * @brief Helper to load an ONNX session and extract I/O names.
     */
    Result<void> load_session(
        Ort::Env& env,
        const std::string& model_path,
        int num_threads,
        std::unique_ptr<Ort::Session>& session,
        std::vector<std::string>& input_names_storage,
        std::vector<std::string>& output_names_storage,
        std::vector<const char*>& input_names,
        std::vector<const char*>& output_names
    ) {
        try {
            Ort::SessionOptions options;
            options.SetIntraOpNumThreads(num_threads);
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            session = std::make_unique<Ort::Session>(env, model_path.c_str(), options);

            Ort::AllocatorWithDefaultOptions allocator;

            // Extract input names
            input_names_storage.clear();
            input_names.clear();
            for (std::size_t i = 0; i < session->GetInputCount(); ++i) {
                auto name = session->GetInputNameAllocated(i, allocator);
                input_names_storage.push_back(name.get());
            }
            for (const auto& name : input_names_storage) {
                input_names.push_back(name.c_str());
            }

            // Extract output names
            output_names_storage.clear();
            output_names.clear();
            for (std::size_t i = 0; i < session->GetOutputCount(); ++i) {
                auto name = session->GetOutputNameAllocated(i, allocator);
                output_names_storage.push_back(name.get());
            }
            for (const auto& name : output_names_storage) {
                output_names.push_back(name.c_str());
            }

            return {};

        } catch (const Ort::Exception& e) {
            return std::unexpected(Error::make(
                ErrorCode::ModelLoadError,
                std::string("Failed to load ") + model_path + ": " + e.what()
            ));
        }
    }
}

Result<void> OpenWakeWordBackend::load_model(std::string_view model_path) {
    config_.wakeword_model_path = std::string(model_path);
    return load_models();
}

Result<void> OpenWakeWordBackend::load_models() {
    // Load melspectrogram model
    if (auto result = load_session(
            *impl_->env,
            config_.melspec_model_path,
            config_.num_threads,
            impl_->melspec_session,
            impl_->melspec_input_names_storage,
            impl_->melspec_output_names_storage,
            impl_->melspec_input_names,
            impl_->melspec_output_names
        ); !result) {
        return result;
    }

    // Load embedding model
    if (auto result = load_session(
            *impl_->env,
            config_.embedding_model_path,
            config_.num_threads,
            impl_->embedding_session,
            impl_->embedding_input_names_storage,
            impl_->embedding_output_names_storage,
            impl_->embedding_input_names,
            impl_->embedding_output_names
        ); !result) {
        return result;
    }

    // Load wake word model
    if (auto result = load_session(
            *impl_->env,
            config_.wakeword_model_path,
            config_.num_threads,
            impl_->wakeword_session,
            impl_->wakeword_input_names_storage,
            impl_->wakeword_output_names_storage,
            impl_->wakeword_input_names,
            impl_->wakeword_output_names
        ); !result) {
        return result;
    }

    // Wake word input is [1, frames, emb_dim]; frames differs between models
    try {
        auto shape = impl_->wakeword_session->GetInputTypeInfo(0)
            .GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() == 3 && shape[1] > 0) {
            config_.history_frames = static_cast<std::uint16_t>(shape[1]);
        }
        if (shape.size() == 3 && shape[2] > 0) {
            config_.embedding_dim = static_cast<std::uint16_t>(shape[2]);
        }
    } catch (const Ort::Exception& e) {
        return std::unexpected(Error::make(
            ErrorCode::ModelLoadError,
            std::string("Cannot read wake word input shape: ") + e.what()
        ));
    }

    reset_buffer();

    models_loaded_ = true;
    return {};
}

Result<FeatureVector> OpenWakeWordBackend::infer(const FeatureMatrix& /*features*/) {
    // Legacy interface - not used for OpenWakeWord
    return std::unexpected(Error::make(
        ErrorCode::InferenceError,
        "Use infer_audio() for OpenWakeWord pipeline"
    ));
}

Result<float> OpenWakeWordBackend::infer_audio(AudioSpan audio) {
    if (!models_loaded_) {
        return std::unexpected(Error::make(
            ErrorCode::InferenceError,
            "Models not loaded"
        ));
    }

    // Step 1: Audio → new mel frames appended to the mel buffer
    if (auto mel_result = run_melspectrogram(audio); !mel_result) {
        return std::unexpected(mel_result.error());
    }

    // Step 2: Last 76 mel frames → Embedding
    auto emb_result = run_embedding();
    if (!emb_result) {
        return std::unexpected(emb_result.error());
    }

    // Step 3: Update embedding buffer
    embedding_buffer_.pop_front();
    embedding_buffer_.push_back(std::move(emb_result.value()));

    // Step 4: Embeddings → Wake word classification.
    // Until the history holds only embeddings of real audio the classifier
    // sees inputs it was never trained on and can spike to ~1.0. That takes
    // ~10 chunks to flush the initial mel frames (8 new frames per chunk)
    // plus history_frames chunks to fill the embedding history.
    constexpr std::size_t kMelFlushChunks = (kEmbeddingWindow + 7) / 8;
    auto confidence = run_wakeword();
    if (confidence && ++predictions_since_reset_ < kMelFlushChunks + config_.history_frames) {
        return 0.0f;
    }
    return confidence;
}

Result<void> OpenWakeWordBackend::run_melspectrogram(AudioSpan audio) {
    try {
        // Prepend the previous tail so the STFT windows straddle chunk borders.
        // The model expects raw int16 values as float32 (no /32768 scaling).
        std::vector<float> audio_float;
        audio_float.reserve(audio_tail_.size() + audio.size());
        for (auto s : audio_tail_) audio_float.push_back(static_cast<float>(s));
        for (auto s : audio) audio_float.push_back(static_cast<float>(s));

        audio_tail_.insert(audio_tail_.end(), audio.begin(), audio.end());
        audio_tail_.erase(audio_tail_.begin(), audio_tail_.end() - kMelOverlapSamples);

        // Input shape: [1, samples]
        std::array<std::int64_t, 2> input_shape = {1, static_cast<std::int64_t>(audio_float.size())};

        auto input_tensor = Ort::Value::CreateTensor<float>(
            *impl_->memory_info,
            audio_float.data(),
            audio_float.size(),
            input_shape.data(),
            input_shape.size()
        );

        auto outputs = impl_->melspec_session->Run(
            Ort::RunOptions{nullptr},
            impl_->melspec_input_names.data(),
            &input_tensor,
            1,
            impl_->melspec_output_names.data(),
            impl_->melspec_output_names.size()
        );

        // Output [1, 1, frames, 32]: 1280 + 480 samples give 8 frames
        const float* output_data = outputs[0].GetTensorData<float>();
        const auto output_size = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        if (output_size % kMelBins != 0) {
            return std::unexpected(Error::make(
                ErrorCode::InferenceError,
                "Melspectrogram output is not a multiple of 32 bins"
            ));
        }

        // Same scaling openWakeWord applies before the embedding model
        for (std::size_t f = 0; f < output_size / kMelBins; ++f) {
            MelFrame frame;
            for (std::size_t b = 0; b < kMelBins; ++b) {
                frame[b] = output_data[f * kMelBins + b] / 10.0f + 2.0f;
            }
            mel_buffer_.push_back(frame);
        }
        while (mel_buffer_.size() > kEmbeddingWindow) {
            mel_buffer_.pop_front();
        }

        return {};

    } catch (const Ort::Exception& e) {
        return std::unexpected(Error::make(
            ErrorCode::InferenceError,
            std::string("Melspectrogram error: ") + e.what()
        ));
    }
}

Result<std::vector<float>> OpenWakeWordBackend::run_embedding() {
    try {
        // Embedding model expects [batch, 76, 32, 1] (frames x mel bins)
        const std::int64_t batch = 1;
        const std::int64_t frames = kEmbeddingWindow;
        const std::int64_t mel_bins = kMelBins;
        const std::int64_t channels = 1;

        std::array<std::int64_t, 4> input_shape = {batch, frames, mel_bins, channels};

        std::vector<float> input_data;
        input_data.reserve(kEmbeddingWindow * kMelBins);
        for (const auto& frame : mel_buffer_) {
            input_data.insert(input_data.end(), frame.begin(), frame.end());
        }

        auto input_tensor = Ort::Value::CreateTensor<float>(
            *impl_->memory_info,
            input_data.data(),
            input_data.size(),
            input_shape.data(),
            input_shape.size()
        );

        auto outputs = impl_->embedding_session->Run(
            Ort::RunOptions{nullptr},
            impl_->embedding_input_names.data(),
            &input_tensor,
            1,
            impl_->embedding_output_names.data(),
            impl_->embedding_output_names.size()
        );

        // Output shape: [1, 1, 1, 96] → flatten to [96]
        const float* output_data = outputs[0].GetTensorData<float>();
        auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
        const auto output_size = output_info.GetElementCount();

        return std::vector<float>(output_data, output_data + output_size);

    } catch (const Ort::Exception& e) {
        return std::unexpected(Error::make(
            ErrorCode::InferenceError,
            std::string("Embedding error: ") + e.what()
        ));
    }
}

Result<float> OpenWakeWordBackend::run_wakeword() {
    try {
        // Wake word model expects [1, 16, 96]
        const std::int64_t batch = 1;
        const std::int64_t frames = static_cast<std::int64_t>(config_.history_frames);
        const std::int64_t emb_dim = static_cast<std::int64_t>(config_.embedding_dim);

        std::array<std::int64_t, 3> input_shape = {batch, frames, emb_dim};

        // Flatten embedding buffer to contiguous array
        std::vector<float> input_data;
        input_data.reserve(static_cast<std::size_t>(frames * emb_dim));
        for (const auto& embedding : embedding_buffer_) {
            input_data.insert(input_data.end(), embedding.begin(), embedding.end());
        }

        // Pad if necessary
        input_data.resize(static_cast<std::size_t>(frames * emb_dim), 0.0f);

        auto input_tensor = Ort::Value::CreateTensor<float>(
            *impl_->memory_info,
            input_data.data(),
            input_data.size(),
            input_shape.data(),
            input_shape.size()
        );

        auto outputs = impl_->wakeword_session->Run(
            Ort::RunOptions{nullptr},
            impl_->wakeword_input_names.data(),
            &input_tensor,
            1,
            impl_->wakeword_output_names.data(),
            impl_->wakeword_output_names.size()
        );

        // Output: [1, 1] → single confidence value
        const float* output_data = outputs[0].GetTensorData<float>();
        float confidence = output_data[0];

        // Apply sigmoid if output is logit
        if (confidence < 0.0f || confidence > 1.0f) {
            confidence = 1.0f / (1.0f + std::exp(-confidence));
        }

        return confidence;

    } catch (const Ort::Exception& e) {
        return std::unexpected(Error::make(
            ErrorCode::InferenceError,
            std::string("Wake word classification error: ") + e.what()
        ));
    }
}

void OpenWakeWordBackend::reset_buffer() {
    // openWakeWord starts the mel buffer as 76 frames of ones
    audio_tail_.assign(kMelOverlapSamples, 0);

    MelFrame ones;
    ones.fill(1.0f);
    mel_buffer_.assign(kEmbeddingWindow, ones);

    embedding_buffer_.assign(config_.history_frames, std::vector<float>(config_.embedding_dim, 0.0f));
    predictions_since_reset_ = 0;
}

std::string_view OpenWakeWordBackend::backend_name() const noexcept {
    return "OpenWakeWord (ONNX Runtime)";
}

bool OpenWakeWordBackend::is_loaded() const noexcept {
    return models_loaded_;
}

const OpenWakeWordConfig& OpenWakeWordBackend::get_config() const noexcept {
    return config_;
}

std::unique_ptr<OpenWakeWordBackend> create_openwakeword_backend(
    const std::string& wakeword_model_path
) {
    namespace fs = std::filesystem;

    OpenWakeWordConfig config;
    config.wakeword_model_path = wakeword_model_path;

    // Assume other models are in same directory
    fs::path model_dir = fs::path(wakeword_model_path).parent_path();
    config.melspec_model_path = (model_dir / "melspectrogram.onnx").string();
    config.embedding_model_path = (model_dir / "embedding_model.onnx").string();

    return std::make_unique<OpenWakeWordBackend>(config);
}

} // namespace wake_word
