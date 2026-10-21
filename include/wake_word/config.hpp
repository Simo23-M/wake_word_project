/**
 * @file config.hpp
 * @brief Configuration structures for the wake word detection system.
 *
 * Defines all configurable parameters for audio capture, MFCC feature
 * extraction, wake word detection, and model inference.
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace wake_word {

/**
 * @brief Audio input configuration parameters.
 *
 * Standard speech recognition settings optimized for wake word detection.
 * Uses 16kHz mono audio which provides sufficient frequency range for
 * human speech while minimizing computational overhead.
 */
struct AudioConfig {
    std::uint32_t sample_rate = 16000;      ///< Sample rate in Hz (16kHz for speech)
    std::uint16_t channels = 1;              ///< Number of audio channels (mono)
    std::uint16_t bits_per_sample = 16;      ///< Bit depth per sample
    std::uint32_t frame_size_ms = 30;        ///< Analysis frame duration in milliseconds
    std::uint32_t hop_size_ms = 10;          ///< Hop between consecutive frames in milliseconds

    /**
     * @brief Calculate the number of samples per analysis frame.
     * @return Number of samples in one frame.
     */
    [[nodiscard]] constexpr std::uint32_t samples_per_frame() const noexcept {
        return (sample_rate * frame_size_ms) / 1000;
    }

    /**
     * @brief Calculate the hop size in samples.
     * @return Number of samples between frame starts.
     */
    [[nodiscard]] constexpr std::uint32_t hop_samples() const noexcept {
        return (sample_rate * hop_size_ms) / 1000;
    }
};

/**
 * @brief MFCC (Mel-Frequency Cepstral Coefficients) extraction configuration.
 *
 * Parameters for converting raw audio into feature vectors suitable for
 * neural network inference. MFCC features capture the spectral envelope
 * of speech signals in a perceptually meaningful way.
 */
struct MfccConfig {
    std::uint16_t num_mfcc_coeffs = 13;      ///< Number of cepstral coefficients to extract
    std::uint16_t num_mel_filters = 40;      ///< Number of triangular mel filterbank channels
    std::uint16_t fft_size = 512;            ///< FFT window size (should be power of 2)
    float pre_emphasis = 0.97f;              ///< Pre-emphasis filter coefficient
    float low_freq_hz = 20.0f;               ///< Lower frequency bound for mel filterbank
    float high_freq_hz = 8000.0f;            ///< Upper frequency bound (Nyquist for 16kHz)
};

/**
 * @brief Wake word detection behavior configuration.
 *
 * Controls detection sensitivity, smoothing, and timing parameters
 * to balance between false positives and missed detections.
 */
struct DetectorConfig {
    std::string_view wake_word = "ok simone"; ///< Target wake word phrase
    float detection_threshold = 0.85f;        ///< Minimum confidence for positive detection [0-1]
    std::uint32_t min_silence_ms = 200;       ///< Cooldown period after detection in milliseconds
    std::uint32_t max_duration_ms = 2000;     ///< Maximum expected wake word duration
    std::uint16_t smoothing_window = 5;       ///< Sliding window size for confidence smoothing
};

/**
 * @brief Neural network model configuration.
 *
 * Settings for loading and running the ONNX inference model.
 */
struct ModelConfig {
    std::string model_path;                  ///< Path to the ONNX model file
    bool use_quantized = true;               ///< Use INT8 quantized model for faster inference
    std::uint16_t num_threads = 2;           ///< Number of threads for parallel inference
};

/**
 * @brief Aggregate configuration for the entire detection system.
 */
struct Config {
    AudioConfig audio;       ///< Audio capture settings
    MfccConfig mfcc;         ///< Feature extraction settings
    DetectorConfig detector; ///< Detection behavior settings
    ModelConfig model;       ///< Model inference settings
};

} // namespace wake_word
