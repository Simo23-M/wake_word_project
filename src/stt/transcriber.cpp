/**
 * @file transcriber.cpp
 * @brief whisper.cpp wrapper implementation.
 */

#include "transcriber.hpp"

#include <whisper.h>

#include <algorithm>
#include <thread>
#include <vector>

namespace wake_word {

Transcriber::Transcriber(const std::string& model_path, std::string language)
    : language_(std::move(language))
{
    ctx_ = whisper_init_from_file_with_params(model_path.c_str(), whisper_context_default_params());
}

Transcriber::~Transcriber() {
    if (ctx_) whisper_free(ctx_);
}

Result<std::string> Transcriber::transcribe(AudioSpan audio) {
    if (!ctx_) {
        return std::unexpected(Error::make(ErrorCode::ModelLoadError, "Whisper model not loaded"));
    }

    std::vector<float> pcm(audio.size());
    for (std::size_t i = 0; i < audio.size(); ++i) pcm[i] = audio[i] / 32768.0f;

    auto params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.language = language_.c_str();
    params.n_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.no_context = true;
    params.single_segment = true;

    if (whisper_full(ctx_, params, pcm.data(), static_cast<int>(pcm.size())) != 0) {
        return std::unexpected(Error::make(ErrorCode::InferenceError, "Whisper transcription failed"));
    }

    std::string text;
    for (int i = 0; i < whisper_full_n_segments(ctx_); ++i) {
        text += whisper_full_get_segment_text(ctx_, i);
    }

    auto first = text.find_first_not_of(" \t\n");
    if (first == std::string::npos) return std::string{};
    return text.substr(first, text.find_last_not_of(" \t\n") - first + 1);
}

} // namespace wake_word
