/**
 * @file mic_check.cpp
 * @brief Microphone and wake word diagnostic.
 *
 * Records from the default input device, prints the level and the raw
 * (unsmoothed) wake word confidence of every 80 ms chunk, and saves the
 * audio to a WAV file for offline analysis.
 *
 * Usage: mic_check [seconds] [wakeword.onnx] [out.wav]
 */

#include "audio/audio_capture.hpp"
#include "inference/openwakeword_backend.hpp"

#include <sndfile.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <print>
#include <string>

using namespace wake_word;

int main(int argc, char* argv[]) {
    const int seconds = argc > 1 ? std::stoi(argv[1]) : 8;
    const std::string model_path = argc > 2 ? argv[2] : "models/alexa_v0.1.onnx";
    const std::string wav_path = argc > 3 ? argv[3] : "data/mic_check.wav";
    constexpr std::size_t kChunk = 1280;

    auto backend = create_openwakeword_backend(model_path);
    if (auto r = backend->load_models(); !r) {
        std::println(stderr, "ERROR: {}", r.error().message);
        return 1;
    }

    AudioConfig config;
    config.frame_size_ms = 80;
    AudioCapture capture(config);

    std::mutex mutex;
    std::condition_variable cv;
    AudioBuffer recorded;
    const std::size_t target = static_cast<std::size_t>(seconds) * config.sample_rate;

    capture.set_callback([&](AudioSpan audio) {
        std::lock_guard lock(mutex);
        recorded.insert(recorded.end(), audio.begin(), audio.end());
        cv.notify_one();
    });
    if (auto r = capture.start(); !r) {
        std::println(stderr, "ERROR: {}", r.error().message);
        return 1;
    }

    std::println("Recording {} s - say the wake word now.", seconds);
    std::println("  time   rms   peak  level                           conf");

    std::size_t processed = 0;
    double max_rms = 0.0;
    float max_conf = 0.0f;
    int max_peak = 0;

    while (processed + kChunk <= target) {
        AudioBuffer chunk;
        {
            std::unique_lock lock(mutex);
            cv.wait(lock, [&] { return recorded.size() >= processed + kChunk; });
            chunk.assign(recorded.begin() + processed, recorded.begin() + processed + kChunk);
        }
        processed += kChunk;

        double sum = 0.0;
        int peak = 0;
        for (auto s : chunk) {
            sum += double(s) * s;
            peak = std::max(peak, std::abs(int(s)));
        }
        const double rms = std::sqrt(sum / kChunk);
        // 0..30 bars on a log scale: 1 bar ≈ 3 dB, full = int16 full scale
        const double db = 20.0 * std::log10(std::max(rms, 1.0) / 32768.0);
        const auto bars = static_cast<std::size_t>(std::clamp((db + 90.0) / 3.0, 0.0, 30.0));

        auto conf = backend->infer_audio(AudioSpan{chunk});
        const float c = conf ? *conf : -1.0f;

        max_rms = std::max(max_rms, rms);
        max_peak = std::max(max_peak, peak);
        max_conf = std::max(max_conf, c);

        std::println("{:5.2f}s {:5.0f} {:6} {:<30}  {:.3f}{}",
            double(processed) / config.sample_rate, rms, peak,
            std::string(bars, '#'), c, c >= 0.5f ? "  <== DETECTED" : "");
    }
    capture.stop();

    SF_INFO info{};
    info.samplerate = static_cast<int>(config.sample_rate);
    info.channels = 1;
    info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    if (SNDFILE* f = sf_open(wav_path.c_str(), SFM_WRITE, &info)) {
        sf_write_short(f, recorded.data(), static_cast<sf_count_t>(processed));
        sf_close(f);
        std::println("\nSaved {}", wav_path);
    } else {
        std::println(stderr, "\nCannot write {}: {}", wav_path, sf_strerror(nullptr));
    }

    std::println("Max RMS {:.0f}, max peak {}, max confidence {:.3f}", max_rms, max_peak, max_conf);
    if (max_peak == 0) {
        std::println("=> Pure silence: the device delivers zeros (wrong input, muted, or privacy switch).");
    } else if (max_rms >= 20000) {
        std::println("=> Signal stuck at full scale: the ALSA capture path is muted/off.");
        std::println("   Fix: amixer -c 0 sset Capture cap 80%  (or alsamixer, F4, space on Capture).");
    } else if (max_rms < 300) {
        std::println("=> Very low level: raise the mic gain (pavucontrol / alsamixer 'Capture', 'Mic Boost').");
    } else if (max_peak >= 32767 && max_rms < 20000) {
        std::println("=> Clipping: lower the mic gain.");
    } else {
        std::println("=> Level looks OK.");
    }
    return 0;
}
