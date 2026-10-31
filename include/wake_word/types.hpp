/**
 * @file types.hpp
 * @brief Core type definitions for the wake word detection system.
 *
 * Provides error handling types, audio data containers, and detection
 * result structures used throughout the codebase.
 */

#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace wake_word {

/**
 * @brief Error codes for system operations.
 */
enum class ErrorCode {
    Success = 0,          ///< Operation completed successfully
    AudioDeviceNotFound,  ///< No audio input device available
    AudioStreamError,     ///< Error opening or reading audio stream
    ModelLoadError,       ///< Failed to load ONNX model
    InferenceError,       ///< Error during neural network inference
    InvalidConfig,        ///< Invalid configuration parameters
    FileNotFound,         ///< Requested file does not exist
    OutOfMemory           ///< Memory allocation failure
};

/**
 * @brief Error information with code and descriptive message.
 */
struct Error {
    ErrorCode code;       ///< Machine-readable error code
    std::string message;  ///< Human-readable error description

    /**
     * @brief Factory method to create an Error instance.
     * @param c Error code.
     * @param msg Optional descriptive message.
     * @return Constructed Error object.
     */
    [[nodiscard]] static Error make(ErrorCode c, std::string msg = "") {
        return Error{c, std::move(msg)};
    }
};

/**
 * @brief Result type for operations that may fail.
 * @tparam T The success value type.
 *
 * Uses std::expected (C++23) for monadic error handling without exceptions.
 */
template<typename T>
using Result = std::expected<T, Error>;

// Audio type aliases
using AudioSample = std::int16_t;              ///< Single 16-bit PCM audio sample
using AudioBuffer = std::vector<AudioSample>;  ///< Owning audio sample container
using AudioSpan = std::span<const AudioSample>;///< Non-owning view into audio data

/**
 * @brief Result of wake word detection for a single analysis window.
 */
struct Detection {
    float confidence;           ///< Detection confidence score [0.0 - 1.0]
    std::uint64_t timestamp_ms; ///< Detection timestamp in milliseconds since epoch
    bool is_detected;           ///< True if confidence exceeds threshold

    /**
     * @brief Boolean conversion for easy conditional checks.
     * @return True if wake word was detected.
     */
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return is_detected;
    }
};

} // namespace wake_word
