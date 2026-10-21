/**
 * @file transcriber.hpp
 * @brief Local speech-to-text using whisper.cpp.
 */

#pragma once

#include "wake_word/types.hpp"

#include <string>

struct whisper_context;

namespace wake_word {

class Transcriber {
public:
    /**
     * @param model_path Path to a ggml whisper model (e.g. models/ggml-base.bin).
     * @param language   Spoken language code, or "auto".
     */
    explicit Transcriber(const std::string& model_path, std::string language = "it");
    ~Transcriber();

    Transcriber(const Transcriber&) = delete;
    Transcriber& operator=(const Transcriber&) = delete;

    [[nodiscard]] bool is_loaded() const noexcept { return ctx_ != nullptr; }

    /// Transcribe 16 kHz mono audio. Returns trimmed text.
    [[nodiscard]] Result<std::string> transcribe(AudioSpan audio);

private:
    whisper_context* ctx_ = nullptr;
    std::string language_;
};

} // namespace wake_word
