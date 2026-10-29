/**
 * @file audio_capture.cpp
 * @brief Implementation of real-time audio capture.
 */

#include "audio_capture.hpp"

#include <portaudio.h>
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace wake_word {

namespace {
    /// Reference count for PortAudio initialization (thread-safe)
    std::atomic<int> pa_init_count{0};

    /**
     * @brief RAII wrapper for PortAudio library initialization.
     *
     * Ensures Pa_Initialize/Pa_Terminate are called exactly once
     * across all AudioCapture instances using reference counting.
     */
    struct PaInitializer {
        PaInitializer() {
            if (pa_init_count.fetch_add(1) == 0) {
                PaError err = Pa_Initialize();
                if (err != paNoError) {
                    pa_init_count.fetch_sub(1);
                    throw std::runtime_error(Pa_GetErrorText(err));
                }
            }
        }

        ~PaInitializer() {
            if (pa_init_count.fetch_sub(1) == 1) {
                Pa_Terminate();
            }
        }
    };

    /// Thread-local PortAudio initializer ensures proper lifecycle
    thread_local PaInitializer pa_init;
}

AudioCapture::AudioCapture(const AudioConfig& config)
    : config_(config)
    , filtered_(config.samples_per_frame())
{
    (void)pa_init; // Trigger PortAudio initialization
}

AudioCapture::~AudioCapture() {
    stop();
}

AudioCapture::AudioCapture(AudioCapture&& other) noexcept
    : config_(other.config_)
    , stream_(std::exchange(other.stream_, nullptr))
    , callback_(std::move(other.callback_))
    , running_(other.running_.load())
    , filtered_(std::move(other.filtered_))
    , dc_prev_in_(other.dc_prev_in_)
    , dc_prev_out_(other.dc_prev_out_)
{
    other.running_.store(false);
}

AudioCapture& AudioCapture::operator=(AudioCapture&& other) noexcept {
    if (this != &other) {
        stop();
        config_ = other.config_;
        stream_ = std::exchange(other.stream_, nullptr);
        callback_ = std::move(other.callback_);
        running_.store(other.running_.load());
        filtered_ = std::move(other.filtered_);
        dc_prev_in_ = other.dc_prev_in_;
        dc_prev_out_ = other.dc_prev_out_;
        other.running_.store(false);
    }
    return *this;
}

Result<void> AudioCapture::start() {
    if (running_.load()) {
        return {}; // Already running
    }

    // Configure input stream parameters
    PaStreamParameters input_params{};
    input_params.device = Pa_GetDefaultInputDevice();

    if (input_params.device == paNoDevice) {
        return std::unexpected(Error::make(
            ErrorCode::AudioDeviceNotFound,
            "No default audio input device found"
        ));
    }

    input_params.channelCount = config_.channels;
    input_params.sampleFormat = paInt16;
    input_params.suggestedLatency = Pa_GetDeviceInfo(input_params.device)->defaultLowInputLatency;
    input_params.hostApiSpecificStreamInfo = nullptr;

    // Reset the DC filter before the callback can run (initial transient ~12 ms)
    dc_prev_in_ = dc_prev_out_ = 0.0f;

    // Open the audio stream
    PaError err = Pa_OpenStream(
        &stream_,
        &input_params,
        nullptr, // No output (input only)
        config_.sample_rate,
        config_.samples_per_frame(),
        paClipOff,
        &AudioCapture::pa_callback,
        this
    );

    if (err != paNoError) {
        return std::unexpected(Error::make(
            ErrorCode::AudioStreamError,
            Pa_GetErrorText(err)
        ));
    }

    // Start streaming. running_ must be set first: the callback returns
    // paComplete (ending the stream for good) if it sees running_ == false.
    running_.store(true);
    err = Pa_StartStream(stream_);
    if (err != paNoError) {
        running_.store(false);
        Pa_CloseStream(stream_);
        stream_ = nullptr;
        return std::unexpected(Error::make(
            ErrorCode::AudioStreamError,
            Pa_GetErrorText(err)
        ));
    }

    return {};
}

void AudioCapture::stop() {
    if (stream_) {
        running_.store(false);
        Pa_StopStream(stream_);
        Pa_CloseStream(stream_);
        stream_ = nullptr;
    }
}

bool AudioCapture::is_running() const noexcept {
    return running_.load();
}

void AudioCapture::set_callback(FrameCallback callback) {
    callback_ = std::move(callback);
}

Result<std::vector<std::string>> AudioCapture::list_devices() {
    (void)pa_init; // Ensure PortAudio is initialized

    std::vector<std::string> devices;
    int device_count = Pa_GetDeviceCount();

    if (device_count < 0) {
        return std::unexpected(Error::make(
            ErrorCode::AudioStreamError,
            Pa_GetErrorText(device_count)
        ));
    }

    // Collect only input-capable devices
    for (int i = 0; i < device_count; ++i) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (info && info->maxInputChannels > 0) {
            devices.emplace_back(info->name);
        }
    }

    return devices;
}

int AudioCapture::pa_callback(
    const void* input,
    void* /*output*/,
    unsigned long frame_count,
    const PaStreamCallbackTimeInfo* /*time_info*/,
    unsigned long /*status_flags*/,
    void* user_data
) {
    auto* capture = static_cast<AudioCapture*>(user_data);

    if (!capture->running_.load()) {
        return paComplete;
    }

    // Forward DC-filtered audio data to user callback
    if (capture->callback_ && input) {
        auto* samples = static_cast<const AudioSample*>(input);
        if (capture->filtered_.size() < frame_count) {
            capture->filtered_.resize(frame_count); // Only if PortAudio changes buffer size
        }
        std::span<AudioSample> out{capture->filtered_.data(), frame_count};
        std::copy_n(samples, frame_count, out.begin());
        capture->remove_dc(out);
        capture->callback_(AudioSpan{out});
    }

    return paContinue;
}

void AudioCapture::remove_dc(std::span<AudioSample> samples) noexcept {
    // y[n] = x[n] - x[n-1] + R * y[n-1]; R = 0.996 → cutoff ≈ 10 Hz @ 16 kHz
    constexpr float kR = 0.996f;
    for (auto& s : samples) {
        const float x = s;
        const float y = x - dc_prev_in_ + kR * dc_prev_out_;
        dc_prev_in_ = x;
        dc_prev_out_ = y;
        s = static_cast<AudioSample>(std::clamp(y, -32768.0f, 32767.0f));
    }
}

} // namespace wake_word
