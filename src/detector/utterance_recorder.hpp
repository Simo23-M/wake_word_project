/**
 * @file utterance_recorder.hpp
 * @brief Records the speech that follows the wake word.
 *
 * Energy-based endpointing: recording ends after `silence_ms` of quiet
 * once speech has started, or after `max_ms` in any case.
 */

#pragma once

#include "wake_word/types.hpp"

#include <cmath>
#include <cstddef>

namespace wake_word {

struct UtteranceRecorder {
    // ponytail: fixed RMS threshold, swap for a real VAD (e.g. Silero) if noisy rooms cause cut-offs
    float silence_rms = 500.0f;        ///< Below this RMS (int16 scale) a chunk counts as silence
    std::size_t silence_ms = 1000;     ///< Trailing silence that ends the utterance
    std::size_t no_speech_ms = 3000;   ///< Give up if nobody speaks within this time
    std::size_t max_ms = 10000;        ///< Hard cap on recording length
    std::size_t sample_rate = 16000;

    AudioBuffer audio;

    /// Append a chunk. Returns true when the utterance is complete.
    bool push(AudioSpan chunk) {
        audio.insert(audio.end(), chunk.begin(), chunk.end());

        double sum = 0.0;
        for (auto s : chunk) sum += double(s) * s;
        bool loud = !chunk.empty() && std::sqrt(sum / chunk.size()) >= silence_rms;

        std::size_t chunk_ms = chunk.size() * 1000 / sample_rate;
        if (loud) { heard_speech = true; quiet_ms = 0; }
        else quiet_ms += chunk_ms;

        std::size_t total_ms = audio.size() * 1000 / sample_rate;
        return total_ms >= max_ms
            || (heard_speech && quiet_ms >= silence_ms)
            || (!heard_speech && total_ms >= no_speech_ms);
    }

    /// True if the recording contains any speech.
    [[nodiscard]] bool has_speech() const noexcept { return heard_speech; }

    void reset() { audio.clear(); heard_speech = false; quiet_ms = 0; }

private:
    bool heard_speech = false;
    std::size_t quiet_ms = 0;
};

} // namespace wake_word
