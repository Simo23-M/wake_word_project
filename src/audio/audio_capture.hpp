/**
 * @file audio_capture.hpp
 * @brief Real-time audio capture using PortAudio.
 *
 * Provides a cross-platform interface for capturing audio from the
 * system's default input device with callback-based frame delivery.
 */

#pragma once

#include "wake_word/config.hpp"
#include "wake_word/types.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <thread>

struct PaStreamCallbackTimeInfo;
typedef void PaStream;

namespace wake_word {

/**
 * @class AudioCapture
 * @brief Manages real-time audio input from the system microphone.
 *
 * Uses PortAudio for cross-platform audio capture. Audio frames are
 * delivered asynchronously via a user-provided callback function.
 *
 * @note Not copyable, but movable. Only one instance should capture
 *       from a device at a time.
 *
 * Example usage:
 * @code
 * AudioConfig config;
 * AudioCapture capture(config);
 * capture.set_callback([](AudioSpan frame) {
 *     // Process audio frame
 * });
 * capture.start();
 * @endcode
 */
class AudioCapture {
public:
    /// Callback type for receiving audio frames
    using FrameCallback = std::function<void(AudioSpan)>;

    /**
     * @brief Construct an audio capture instance.
     * @param config Audio configuration parameters.
     */
    explicit AudioCapture(const AudioConfig& config);

    /**
     * @brief Destructor. Stops capture if running.
     */
    ~AudioCapture();

    AudioCapture(const AudioCapture&) = delete;
    AudioCapture& operator=(const AudioCapture&) = delete;

    /**
     * @brief Move constructor.
     * @param other Source instance (will be invalidated).
     */
    AudioCapture(AudioCapture&&) noexcept;

    /**
     * @brief Move assignment operator.
     * @param other Source instance (will be invalidated).
     * @return Reference to this instance.
     */
    AudioCapture& operator=(AudioCapture&&) noexcept;

    /**
     * @brief Start capturing audio from the default input device.
     * @return Success or error with description.
     */
    [[nodiscard]] Result<void> start();

    /**
     * @brief Stop audio capture.
     *
     * Safe to call multiple times or when not running.
     */
    void stop();

    /**
     * @brief Check if capture is currently active.
     * @return True if capturing audio.
     */
    [[nodiscard]] bool is_running() const noexcept;

    /**
     * @brief Set the callback for receiving audio frames.
     * @param callback Function to call with each audio frame.
     *
     * @warning Callback is invoked from the audio thread. Keep
     *          processing minimal to avoid buffer underruns.
     */
    void set_callback(FrameCallback callback);

    /**
     * @brief Enumerate available audio input devices.
     * @return List of device names or error.
     */
    [[nodiscard]] static Result<std::vector<std::string>> list_devices();

private:
    /**
     * @brief PortAudio stream callback (static).
     *
     * Called by PortAudio when audio data is available.
     */
    static int pa_callback(
        const void* input,
        void* output,
        unsigned long frame_count,
        const PaStreamCallbackTimeInfo* time_info,
        unsigned long status_flags,
        void* user_data
    );

    AudioConfig config_;              ///< Audio configuration
    PaStream* stream_ = nullptr;      ///< PortAudio stream handle
    FrameCallback callback_;          ///< User callback for frames
    std::atomic<bool> running_{false};///< Capture state flag
    AudioBuffer ring_buffer_;         ///< Internal circular buffer
    std::size_t write_pos_ = 0;       ///< Ring buffer write position
};

} // namespace wake_word
