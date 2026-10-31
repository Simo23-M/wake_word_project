/**
 * @file config.hpp
 * @brief Configuration structures for the wake word detection system.
 *
 * Defines the configurable parameters for audio capture.
 */

#pragma once

#include <cstdint>

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

} // namespace wake_word
