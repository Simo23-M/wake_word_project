/**
 * @file openwakeword_backend.hpp
 * @brief OpenWakeWord-compatible inference backend.
 *
 * Implements the three-model pipeline used by OpenWakeWord:
 * melspectrogram → embedding → wake word classification.
 */

#pragma once

#include "wake_word/types.hpp"

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace wake_word {

/**
 * @brief Configuration for OpenWakeWord pipeline.
 */
struct OpenWakeWordConfig {
    std::string melspec_model_path;      ///< Path to melspectrogram.onnx
    std::string embedding_model_path;    ///< Path to embedding_model.onnx
    std::string wakeword_model_path;     ///< Path to wake word model (e.g., alexa_v0.1.onnx)

    std::uint32_t sample_rate = 16000;   ///< Audio sample rate (fixed)
    std::uint32_t chunk_samples = 1280;  ///< Samples per chunk (80ms @ 16kHz)
    std::uint16_t embedding_dim = 96;    ///< Embedding vector dimension
    std::uint16_t history_frames = 16;   ///< Number of embedding frames for classification
    std::uint16_t num_threads = 2;       ///< ONNX Runtime thread count
};

/**
 * @class OpenWakeWordBackend
 * @brief Three-model inference pipeline for OpenWakeWord compatibility.
 *
 * Pipeline flow:
 * @code
 * Audio chunk [1280 samples]
 *     │
 *     ▼
 * [Melspectrogram Model] ──► mel features [time, 1, ?, 32]
 *     │
 *     ▼
 * [Embedding Model] ──► embedding [1, 1, 1, 96]
 *     │
 *     ▼
 * [Embedding Buffer] ──► stores last 16 embeddings
 *     │
 *     ▼
 * [Wake Word Model] ──► confidence [0.0 - 1.0]
 * @endcode
 *
 * The embedding buffer maintains temporal context required for
 * accurate wake word detection across multiple audio frames.
 */
class OpenWakeWordBackend {
public:
    /**
     * @brief Construct OpenWakeWord backend with given configuration.
     * @param config Pipeline configuration including model paths.
     */
    explicit OpenWakeWordBackend(const OpenWakeWordConfig& config);

    /**
     * @brief Destructor. Releases all ONNX Runtime resources.
     */
    ~OpenWakeWordBackend();

    OpenWakeWordBackend(const OpenWakeWordBackend&) = delete;
    OpenWakeWordBackend& operator=(const OpenWakeWordBackend&) = delete;
    OpenWakeWordBackend(OpenWakeWordBackend&&) noexcept;
    OpenWakeWordBackend& operator=(OpenWakeWordBackend&&) noexcept;

    /**
     * @brief Load all three models in the pipeline.
     * @return Success or error with description.
     *
     * Loads melspectrogram, embedding, and wake word models.
     * Paths are taken from the configuration.
     */
    [[nodiscard]] Result<void> load_models();

    /**
     * @brief Run full inference pipeline on raw audio.
     * @param audio Raw audio samples (should be chunk_samples length).
     * @return Confidence score [0.0 - 1.0] or error.
     *
     * Performs: audio → melspec (streaming) → embedding of the last 76 mel
     * frames → embedding history → classification. Returns 0 for ~2 s after
     * a reset, until mel and embedding history hold only real audio.
     */
    [[nodiscard]] Result<float> infer_audio(AudioSpan audio);

    /**
     * @brief Reset audio, mel and embedding history.
     *
     * Call after detection to start fresh for next wake word.
     */
    void reset_buffer();

    /**
     * @brief Check if all models are loaded.
     * @return True if ready for inference.
     */
    [[nodiscard]] bool is_loaded() const noexcept;

private:
    /// Mel bins produced by melspectrogram.onnx
    static constexpr std::size_t kMelBins = 32;
    /// Mel frames consumed by one embedding (~775 ms)
    static constexpr std::size_t kEmbeddingWindow = 76;
    /// Samples of the previous chunk prepended to each new one (3 hops of 160)
    static constexpr std::size_t kMelOverlapSamples = 480;
    using MelFrame = std::array<float, kMelBins>;

    /**
     * @brief Run melspectrogram model on the overlap + new audio.
     * @param audio New raw audio samples.
     * @return Success or error. New frames are appended to the mel buffer.
     */
    [[nodiscard]] Result<void> run_melspectrogram(AudioSpan audio);

    /**
     * @brief Run embedding model on the last kEmbeddingWindow mel frames.
     * @return Embedding vector or error.
     */
    [[nodiscard]] Result<std::vector<float>> run_embedding();

    /**
     * @brief Run wake word classification.
     * @return Confidence score or error.
     *
     * Uses the embedding buffer as input.
     */
    [[nodiscard]] Result<float> run_wakeword();

    /// PIMPL for ONNX Runtime internals
    struct Impl;
    std::unique_ptr<Impl> impl_;

    OpenWakeWordConfig config_;

    /// Last kMelOverlapSamples of audio, so mel frames stay continuous across chunks
    AudioBuffer audio_tail_;

    /// Most recent mel frames (at least kEmbeddingWindow)
    std::deque<MelFrame> mel_buffer_;

    /// Circular buffer of recent embeddings
    std::deque<std::vector<float>> embedding_buffer_;

    std::size_t predictions_since_reset_ = 0;
    bool models_loaded_ = false;
};

/**
 * @brief Create OpenWakeWord backend with default model paths.
 * @param wakeword_model_path Path to the wake word model.
 * @return Configured backend instance.
 *
 * Assumes melspectrogram.onnx and embedding_model.onnx are in
 * the same directory as the wake word model.
 */
[[nodiscard]] std::unique_ptr<OpenWakeWordBackend> create_openwakeword_backend(
    const std::string& wakeword_model_path
);

} // namespace wake_word
