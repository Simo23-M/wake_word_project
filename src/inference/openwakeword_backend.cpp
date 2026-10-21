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

    // Initialize embedding buffer with zeros
    embedding_buffer_.clear();
    std::vector<float> zero_embedding(config_.embedding_dim, 0.0f);
    for (std::uint16_t i = 0; i < config_.history_frames; ++i) {
        embedding_buffer_.push_back(zero_embedding);
    }

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

    // Step 1: Audio → Melspectrogram
    auto mel_result = run_melspectrogram(audio);
    if (!mel_result) {
        return std::unexpected(mel_result.error());
    }

    // Step 2: Melspectrogram → Embedding
    auto emb_result = run_embedding(mel_result.value());
    if (!emb_result) {
        return std::unexpected(emb_result.error());
    }

    // Step 3: Update embedding buffer
    embedding_buffer_.pop_front();
    embedding_buffer_.push_back(emb_result.value());

    // Step 4: Embeddings → Wake word classification
    return run_wakeword();
}

Result<std::vector<float>> OpenWakeWordBackend::run_melspectrogram(AudioSpan audio) {
    try {
        // Convert int16 audio to float32 and normalize
        std::vector<float> audio_float(audio.size());
        for (std::size_t i = 0; i < audio.size(); ++i) {
            audio_float[i] = static_cast<float>(audio[i]) / 32768.0f;
        }

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

        // Extract output
        const float* output_data = outputs[0].GetTensorData<float>();
        auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
        auto output_shape = output_info.GetShape();
        const auto output_size = output_info.GetElementCount();

        return std::vector<float>(output_data, output_data + output_size);

    } catch (const Ort::Exception& e) {
        return std::unexpected(Error::make(
            ErrorCode::InferenceError,
            std::string("Melspectrogram error: ") + e.what()
        ));
    }
}

Result<std::vector<float>> OpenWakeWordBackend::run_embedding(const std::vector<float>& mel_features) {
    try {
        // Embedding model expects [batch, 76, 32, 1]
        // We need to reshape mel features appropriately
        // OpenWakeWord uses 76 mel frames with 32 mel bins

        const std::int64_t batch = 1;
        const std::int64_t frames = 76;
        const std::int64_t mel_bins = 32;
        const std::int64_t channels = 1;

        std::array<std::int64_t, 4> input_shape = {batch, frames, mel_bins, channels};
        const auto expected_size = static_cast<std::size_t>(batch * frames * mel_bins * channels);

        // Pad or truncate to expected size
        std::vector<float> input_data(expected_size, 0.0f);
        const auto copy_size = std::min(mel_features.size(), expected_size);
        std::copy_n(mel_features.begin(), copy_size, input_data.begin());

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
    std::vector<float> zero_embedding(config_.embedding_dim, 0.0f);
    for (auto& emb : embedding_buffer_) {
        emb = zero_embedding;
    }
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
