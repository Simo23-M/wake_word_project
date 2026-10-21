/**
 * @file mfcc_extractor.cpp
 * @brief Implementation of MFCC feature extraction.
 */

#include "mfcc_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>

namespace wake_word {

MfccExtractor::MfccExtractor(const AudioConfig& audio_config, const MfccConfig& mfcc_config)
    : audio_config_(audio_config)
    , mfcc_config_(mfcc_config)
{
    init_hamming_window();
    init_mel_filterbank();
    init_dct_matrix();
}

void MfccExtractor::init_hamming_window() {
    const auto size = mfcc_config_.fft_size;
    hamming_window_.resize(size);

    // Hamming: w[n] = 0.54 - 0.46 * cos(2*pi*n/(N-1))
    for (std::size_t i = 0; i < size; ++i) {
        hamming_window_[i] = 0.54f - 0.46f * std::cos(
            2.0f * std::numbers::pi_v<float> * static_cast<float>(i) / static_cast<float>(size - 1)
        );
    }
}

void MfccExtractor::init_mel_filterbank() {
    const auto num_filters = mfcc_config_.num_mel_filters;
    const auto fft_size = mfcc_config_.fft_size;
    const auto sample_rate = audio_config_.sample_rate;

    // Convert frequency bounds to mel scale
    const float low_mel = hz_to_mel(mfcc_config_.low_freq_hz);
    const float high_mel = hz_to_mel(mfcc_config_.high_freq_hz);

    // Create equally spaced points in mel scale
    std::vector<float> mel_points(num_filters + 2);
    for (std::size_t i = 0; i < mel_points.size(); ++i) {
        mel_points[i] = low_mel + static_cast<float>(i) * (high_mel - low_mel) /
                        static_cast<float>(num_filters + 1);
    }

    // Map mel points to FFT bin indices
    std::vector<std::size_t> bin_points(mel_points.size());
    for (std::size_t i = 0; i < mel_points.size(); ++i) {
        float hz = mel_to_hz(mel_points[i]);
        bin_points[i] = static_cast<std::size_t>(
            std::floor((fft_size + 1) * hz / static_cast<float>(sample_rate))
        );
    }

    // Build triangular filterbank
    mel_filterbank_.resize(num_filters);
    const auto spectrum_size = fft_size / 2 + 1;

    for (std::size_t m = 0; m < num_filters; ++m) {
        mel_filterbank_[m].resize(spectrum_size, 0.0f);

        // Rising edge of triangle
        for (std::size_t k = bin_points[m]; k < bin_points[m + 1]; ++k) {
            if (k < spectrum_size) {
                mel_filterbank_[m][k] = static_cast<float>(k - bin_points[m]) /
                                        static_cast<float>(bin_points[m + 1] - bin_points[m]);
            }
        }

        // Falling edge of triangle
        for (std::size_t k = bin_points[m + 1]; k < bin_points[m + 2]; ++k) {
            if (k < spectrum_size) {
                mel_filterbank_[m][k] = static_cast<float>(bin_points[m + 2] - k) /
                                        static_cast<float>(bin_points[m + 2] - bin_points[m + 1]);
            }
        }
    }
}

void MfccExtractor::init_dct_matrix() {
    const auto num_coeffs = mfcc_config_.num_mfcc_coeffs;
    const auto num_filters = mfcc_config_.num_mel_filters;

    // DCT-II: C[k,n] = cos(pi * k * (n + 0.5) / N)
    dct_matrix_.resize(num_coeffs);
    for (std::size_t k = 0; k < num_coeffs; ++k) {
        dct_matrix_[k].resize(num_filters);
        for (std::size_t n = 0; n < num_filters; ++n) {
            dct_matrix_[k][n] = std::cos(
                std::numbers::pi_v<float> * static_cast<float>(k) *
                (static_cast<float>(n) + 0.5f) / static_cast<float>(num_filters)
            );
        }
    }
}

float MfccExtractor::hz_to_mel(float hz) {
    // O'Shaughnessy formula: mel = 2595 * log10(1 + hz/700)
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float MfccExtractor::mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

void MfccExtractor::apply_pre_emphasis(std::vector<float>& signal) const {
    // Process in reverse to avoid overwriting needed values
    for (std::size_t i = signal.size() - 1; i > 0; --i) {
        signal[i] -= mfcc_config_.pre_emphasis * signal[i - 1];
    }
}

void MfccExtractor::apply_hamming_window(std::vector<float>& frame) const {
    const auto size = std::min(frame.size(), hamming_window_.size());
    for (std::size_t i = 0; i < size; ++i) {
        frame[i] *= hamming_window_[i];
    }
}

std::vector<float> MfccExtractor::compute_power_spectrum(const std::vector<float>& frame) const {
    const auto fft_size = mfcc_config_.fft_size;
    const auto spectrum_size = fft_size / 2 + 1;

    // Direct DFT computation (for production, use FFTW or similar)
    std::vector<float> power_spectrum(spectrum_size);

    for (std::size_t k = 0; k < spectrum_size; ++k) {
        float real = 0.0f;
        float imag = 0.0f;

        for (std::size_t n = 0; n < fft_size && n < frame.size(); ++n) {
            float angle = 2.0f * std::numbers::pi_v<float> * static_cast<float>(k * n) /
                          static_cast<float>(fft_size);
            real += frame[n] * std::cos(angle);
            imag -= frame[n] * std::sin(angle);
        }

        // Power = |X[k]|^2 / N
        power_spectrum[k] = (real * real + imag * imag) / static_cast<float>(fft_size);
    }

    return power_spectrum;
}

std::vector<float> MfccExtractor::apply_mel_filterbank(const std::vector<float>& power_spectrum) const {
    std::vector<float> mel_energies(mel_filterbank_.size());

    for (std::size_t m = 0; m < mel_filterbank_.size(); ++m) {
        float energy = 0.0f;
        const auto& filter = mel_filterbank_[m];
        const auto size = std::min(filter.size(), power_spectrum.size());

        for (std::size_t k = 0; k < size; ++k) {
            energy += filter[k] * power_spectrum[k];
        }

        // Log compression with floor to avoid log(0)
        mel_energies[m] = std::log(std::max(energy, 1e-10f));
    }

    return mel_energies;
}

FeatureVector MfccExtractor::apply_dct(const std::vector<float>& mel_energies) const {
    FeatureVector mfcc(dct_matrix_.size());

    for (std::size_t k = 0; k < dct_matrix_.size(); ++k) {
        float sum = 0.0f;
        const auto size = std::min(dct_matrix_[k].size(), mel_energies.size());

        for (std::size_t n = 0; n < size; ++n) {
            sum += dct_matrix_[k][n] * mel_energies[n];
        }

        mfcc[k] = sum;
    }

    return mfcc;
}

FeatureVector MfccExtractor::extract(AudioSpan audio) const {
    std::vector<float> frame(mfcc_config_.fft_size, 0.0f);

    // Convert 16-bit PCM to normalized float [-1, 1]
    const auto copy_size = std::min(audio.size(), frame.size());
    for (std::size_t i = 0; i < copy_size; ++i) {
        frame[i] = static_cast<float>(audio[i]) / 32768.0f;
    }

    // Apply MFCC pipeline
    apply_pre_emphasis(frame);
    apply_hamming_window(frame);

    auto power_spectrum = compute_power_spectrum(frame);
    auto mel_energies = apply_mel_filterbank(power_spectrum);

    return apply_dct(mel_energies);
}

FeatureMatrix MfccExtractor::extract_batch(AudioSpan audio) const {
    const auto frame_samples = audio_config_.samples_per_frame();
    const auto hop_samples = audio_config_.hop_samples();

    if (audio.size() < frame_samples) {
        return {}; // Not enough samples for even one frame
    }

    // Calculate number of frames with overlap
    const auto num_frames = 1 + (audio.size() - frame_samples) / hop_samples;
    FeatureMatrix features;
    features.reserve(num_frames);

    for (std::size_t i = 0; i < num_frames; ++i) {
        const auto offset = i * hop_samples;
        const auto frame_end = std::min(offset + frame_samples, audio.size());
        features.push_back(extract(audio.subspan(offset, frame_end - offset)));
    }

    return features;
}

} // namespace wake_word
