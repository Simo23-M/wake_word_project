/**
 * @file wake_word_detector.hpp
 * @brief Main wake word detection orchestrator.
 *
 * Integrates audio capture and OpenWakeWord inference pipeline
 * into a complete wake word detection system.
 */

#pragma once

#include "wake_word/config.hpp"
#include "wake_word/types.hpp"
#include "audio/audio_capture.hpp"
#include "inference/openwakeword_backend.hpp"
#include "detector/utterance_recorder.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace wake_word {

/**
 * @class WakeWordDetector
 * @brief Orchestrates the complete wake word detection pipeline.
 *
 * Architecture:
 * @code
 * Microphone → AudioCapture → Ring Buffer → OpenWakeWord Pipeline → Detection
 *                                                    │
 *                                    ┌───────────────┼───────────────┐
 *                                    ▼               ▼               ▼
 *                              Melspectrogram → Embedding → Wake Word Model
 * @endcode
 *
 * The detector processes audio in 80ms chunks (1280 samples @ 16kHz)
 * as required by OpenWakeWord models.
 */
class WakeWordDetector {
public:
    /// User callback for detection events
    using Callback = std::function<void(const Detection&)>;

    /// User callback for the speech recorded after the wake word
    using UtteranceCallback = std::function<void(AudioSpan)>;

    /**
     * @brief Construct detector with model paths.
     * @param wakeword_model_path Path to wake word .onnx model.
     * @param threshold Detection threshold [0.0 - 1.0].
     */
    explicit WakeWordDetector(
        const std::string& wakeword_model_path,
        float threshold = 0.5f
    );

    /**
     * @brief Construct detector with full OpenWakeWord configuration.
     * @param oww_config OpenWakeWord pipeline configuration.
     * @param threshold Detection threshold [0.0 - 1.0].
     */
    explicit WakeWordDetector(
        const OpenWakeWordConfig& oww_config,
        float threshold = 0.5f
    );

    /**
     * @brief Destructor. Stops detection if running.
     */
    ~WakeWordDetector();

    WakeWordDetector(const WakeWordDetector&) = delete;
    WakeWordDetector& operator=(const WakeWordDetector&) = delete;

    /**
     * @brief Initialize all components and load models.
     * @return Success or error with description.
     */
    [[nodiscard]] Result<void> initialize();

    /**
     * @brief Start continuous wake word detection.
     * @return Success or error with description.
     */
    [[nodiscard]] Result<void> start();

    /**
     * @brief Stop detection and release resources.
     */
    void stop();

    /**
     * @brief Check if detection is currently active.
     * @return True if listening for wake word.
     */
    [[nodiscard]] bool is_running() const noexcept;

    /**
     * @brief Set callback for detection events.
     * @param callback Function to call on detection.
     */
    void set_detection_callback(Callback callback);

    /**
     * @brief Set callback receiving the speech that follows the wake word.
     * @param callback Function called (on the detection thread) with the recorded audio.
     */
    void set_utterance_callback(UtteranceCallback callback);

    /**
     * @brief Adjust detection sensitivity at runtime.
     * @param threshold New threshold [0.0 - 1.0].
     */
    void set_threshold(float threshold);

    /**
     * @brief Get current detection threshold.
     * @return Threshold value [0.0 - 1.0].
     */
    [[nodiscard]] float get_threshold() const noexcept;

private:
    /**
     * @brief Audio callback - buffers incoming samples.
     * @param audio Incoming audio frame.
     */
    void on_audio(AudioSpan audio);

    /**
     * @brief Detection thread main loop.
     */
    void detection_loop();

    OpenWakeWordConfig oww_config_;
    AudioConfig audio_config_;

    std::unique_ptr<AudioCapture> audio_capture_;
    std::unique_ptr<OpenWakeWordBackend> backend_;

    Callback detection_callback_;
    UtteranceCallback utterance_callback_;

    UtteranceRecorder recorder_;
    bool recording_ = false;                   ///< Detection thread only

    std::mutex buffer_mutex_;
    AudioBuffer audio_buffer_;                 ///< Accumulator for 80ms chunks
    std::size_t buffer_pos_ = 0;

    std::atomic<bool> running_{false};
    std::atomic<float> threshold_;

    std::jthread detection_thread_;

    /// Smoothing window for confidence scores
    std::vector<float> confidence_history_;
    std::size_t history_index_ = 0;
    static constexpr std::size_t kSmoothingWindow = 3;

    /// Minimum samples needed for detection (80ms @ 16kHz)
    static constexpr std::size_t kChunkSamples = 1280;
};

} // namespace wake_word
