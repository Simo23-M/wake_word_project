/**
 * @file mfcc_extractor.hpp
 * @brief MFCC (Mel-Frequency Cepstral Coefficients) feature extraction.
 *
 * Converts raw audio samples into MFCC feature vectors suitable for
 * speech recognition neural networks. The extraction pipeline follows
 * standard speech processing conventions.
 */

#pragma once

#include "wake_word/config.hpp"
#include "wake_word/types.hpp"

#include <complex>
#include <vector>

namespace wake_word {

/**
 * @class MfccExtractor
 * @brief Extracts MFCC features from audio signals.
 *
 * Processing pipeline:
 * 1. Pre-emphasis filtering (boost high frequencies)
 * 2. Hamming windowing (reduce spectral leakage)
 * 3. FFT to obtain power spectrum
 * 4. Mel filterbank application
 * 5. Log compression
 * 6. DCT to obtain cepstral coefficients
 *
 * All filter coefficients are pre-computed at construction time
 * for efficient runtime extraction.
 */
class MfccExtractor {
public:
    /**
     * @brief Construct an MFCC extractor with given configuration.
     * @param audio_config Audio parameters (sample rate, frame size).
     * @param mfcc_config MFCC parameters (coefficients, filters, FFT size).
     */
    explicit MfccExtractor(const AudioConfig& audio_config, const MfccConfig& mfcc_config);

    /**
     * @brief Extract MFCC features from a single audio frame.
     * @param audio Audio samples (should match frame_size from config).
     * @return Feature vector with num_mfcc_coeffs elements.
     */
    [[nodiscard]] FeatureVector extract(AudioSpan audio) const;

    /**
     * @brief Extract MFCC features from multiple overlapping frames.
     * @param audio Audio samples spanning multiple frames.
     * @return Matrix of features (frames x coefficients).
     *
     * Frames are extracted with overlap according to hop_size configuration.
     */
    [[nodiscard]] FeatureMatrix extract_batch(AudioSpan audio) const;

private:
    /**
     * @brief Apply first-order pre-emphasis filter.
     * @param signal Audio signal (modified in-place).
     *
     * Amplifies high frequencies to balance the spectrum.
     * Formula: y[n] = x[n] - alpha * x[n-1]
     */
    void apply_pre_emphasis(std::vector<float>& signal) const;

    /**
     * @brief Apply Hamming window to a frame.
     * @param frame Audio frame (modified in-place).
     *
     * Reduces spectral leakage at frame boundaries.
     */
    void apply_hamming_window(std::vector<float>& frame) const;

    /**
     * @brief Compute power spectrum via DFT.
     * @param frame Windowed audio frame.
     * @return Power spectrum (magnitude squared).
     */
    [[nodiscard]] std::vector<float> compute_power_spectrum(const std::vector<float>& frame) const;

    /**
     * @brief Apply triangular mel filterbank to power spectrum.
     * @param power_spectrum Input power spectrum.
     * @return Log mel energies for each filter.
     */
    [[nodiscard]] std::vector<float> apply_mel_filterbank(const std::vector<float>& power_spectrum) const;

    /**
     * @brief Apply DCT-II to obtain cepstral coefficients.
     * @param mel_energies Log mel filterbank energies.
     * @return MFCC feature vector.
     */
    [[nodiscard]] FeatureVector apply_dct(const std::vector<float>& mel_energies) const;

    /**
     * @brief Convert frequency from Hz to mel scale.
     * @param hz Frequency in Hertz.
     * @return Frequency in mel scale.
     */
    [[nodiscard]] static float hz_to_mel(float hz);

    /**
     * @brief Convert frequency from mel scale to Hz.
     * @param mel Frequency in mel scale.
     * @return Frequency in Hertz.
     */
    [[nodiscard]] static float mel_to_hz(float mel);

    /// Initialize Hamming window coefficients
    void init_hamming_window();

    /// Initialize triangular mel filterbank
    void init_mel_filterbank();

    /// Initialize DCT-II matrix
    void init_dct_matrix();

    AudioConfig audio_config_;                    ///< Audio parameters
    MfccConfig mfcc_config_;                      ///< MFCC parameters
    std::vector<float> hamming_window_;           ///< Pre-computed window
    std::vector<std::vector<float>> mel_filterbank_; ///< Triangular filters
    std::vector<std::vector<float>> dct_matrix_;  ///< DCT coefficients
};

} // namespace wake_word
