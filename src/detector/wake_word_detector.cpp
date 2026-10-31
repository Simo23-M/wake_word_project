/**
 * @file wake_word_detector.cpp
 * @brief Implementation of the main detection orchestrator.
 */

#include "wake_word_detector.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <numeric>

namespace wake_word {

WakeWordDetector::WakeWordDetector(
    const std::string& wakeword_model_path,
    float threshold
)
    : threshold_(threshold)
    , confidence_history_(kSmoothingWindow, 0.0f)
{
    namespace fs = std::filesystem;

    // Configure OpenWakeWord paths
    fs::path model_dir = fs::path(wakeword_model_path).parent_path();
    oww_config_.wakeword_model_path = wakeword_model_path;
    oww_config_.melspec_model_path = (model_dir / "melspectrogram.onnx").string();
    oww_config_.embedding_model_path = (model_dir / "embedding_model.onnx").string();

    // Audio config for OpenWakeWord compatibility
    audio_config_.sample_rate = 16000;
    audio_config_.channels = 1;
    audio_config_.bits_per_sample = 16;
    audio_config_.frame_size_ms = 80;  // 1280 samples
    audio_config_.hop_size_ms = 80;    // Non-overlapping

    // Pre-allocate audio buffer
    audio_buffer_.resize(kChunkSamples * 4);  // Buffer 4 chunks
}

WakeWordDetector::WakeWordDetector(
    const OpenWakeWordConfig& oww_config,
    float threshold
)
    : oww_config_(oww_config)
    , threshold_(threshold)
    , confidence_history_(kSmoothingWindow, 0.0f)
{
    audio_config_.sample_rate = oww_config.sample_rate;
    audio_config_.channels = 1;
    audio_config_.bits_per_sample = 16;
    audio_config_.frame_size_ms = 80;
    audio_config_.hop_size_ms = 80;

    audio_buffer_.resize(kChunkSamples * 4);
}

WakeWordDetector::~WakeWordDetector() {
    stop();
}

Result<void> WakeWordDetector::initialize() {
    // Create audio capture
    audio_capture_ = std::make_unique<AudioCapture>(audio_config_);

    // Create and load OpenWakeWord backend
    backend_ = std::make_unique<OpenWakeWordBackend>(oww_config_);
    if (auto result = backend_->load_models(); !result) {
        return result;
    }

    // Wire audio callback
    audio_capture_->set_callback([this](AudioSpan audio) {
        on_audio(audio);
    });

    return {};
}

Result<void> WakeWordDetector::start() {
    if (running_.load()) {
        return {};
    }

    // Start audio capture
    if (auto result = audio_capture_->start(); !result) {
        return result;
    }

    running_.store(true);
    buffer_pos_ = 0;

    // Launch detection thread
    detection_thread_ = std::jthread([this](std::stop_token stoken) {
        while (!stoken.stop_requested() && running_.load()) {
            detection_loop();
        }
    });

    return {};
}

void WakeWordDetector::stop() {
    running_.store(false);

    if (detection_thread_.joinable()) {
        detection_thread_.request_stop();
        detection_thread_.join();
    }

    if (audio_capture_) {
        audio_capture_->stop();
    }
}

bool WakeWordDetector::is_running() const noexcept {
    return running_.load();
}

void WakeWordDetector::set_detection_callback(Callback callback) {
    detection_callback_ = std::move(callback);
}

void WakeWordDetector::set_utterance_callback(UtteranceCallback callback) {
    utterance_callback_ = std::move(callback);
}

void WakeWordDetector::set_threshold(float threshold) {
    threshold_.store(std::clamp(threshold, 0.0f, 1.0f));
}

float WakeWordDetector::get_threshold() const noexcept {
    return threshold_.load();
}

void WakeWordDetector::on_audio(AudioSpan audio) {
    std::lock_guard lock(buffer_mutex_);

    // Append to buffer
    for (auto sample : audio) {
        if (buffer_pos_ < audio_buffer_.size()) {
            audio_buffer_[buffer_pos_++] = sample;
        }
    }
}

void WakeWordDetector::detection_loop() {
    using namespace std::chrono_literals;

    // Check if we have enough samples
    AudioBuffer chunk;
    bool ready = false;
    {
        std::lock_guard lock(buffer_mutex_);
        ready = buffer_pos_ >= kChunkSamples;
    }
    if (!ready) {
        // Sleep without the lock, or the audio callback starves on it
        std::this_thread::sleep_for(10ms);
        return;
    }
    {
        std::lock_guard lock(buffer_mutex_);

        // Extract chunk
        chunk.assign(audio_buffer_.begin(), audio_buffer_.begin() + kChunkSamples);

        // Shift remaining data
        std::copy(
            audio_buffer_.begin() + kChunkSamples,
            audio_buffer_.begin() + buffer_pos_,
            audio_buffer_.begin()
        );
        buffer_pos_ -= kChunkSamples;
    }

    // After a detection, record the following speech instead of listening
    if (recording_) {
        if (recorder_.push(AudioSpan{chunk})) {
            if (recorder_.has_speech() && utterance_callback_) {
                utterance_callback_(AudioSpan{recorder_.audio});
            }
            recorder_.reset();
            backend_->reset_buffer();
            recording_ = false;
        }
        return;
    }

    // Run inference
    auto confidence_result = backend_->infer_audio(AudioSpan{chunk});
    if (!confidence_result) {
        return;
    }

    float raw_confidence = confidence_result.value();

    // Update smoothing window
    confidence_history_[history_index_] = raw_confidence;
    history_index_ = (history_index_ + 1) % confidence_history_.size();

    // Compute smoothed confidence
    float smoothed_confidence = std::accumulate(
        confidence_history_.begin(),
        confidence_history_.end(),
        0.0f
    ) / static_cast<float>(confidence_history_.size());

    // Check threshold
    if (smoothed_confidence >= threshold_.load()) {
        Detection detection{};
        detection.confidence = smoothed_confidence;
        detection.timestamp_ms = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()
            ).count()
        );
        detection.is_detected = true;

        // Notify callback
        if (detection_callback_) {
            detection_callback_(detection);
        }

        // Reset state after detection
        backend_->reset_buffer();
        std::ranges::fill(confidence_history_, 0.0f);

        // Record what follows (also acts as re-trigger cooldown)
        recording_ = true;
    }
}

} // namespace wake_word
