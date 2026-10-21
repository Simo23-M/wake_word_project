/**
 * @file test_mfcc.cpp
 * @brief Unit tests for MFCC feature extraction.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "features/mfcc_extractor.hpp"

using namespace wake_word;
using Catch::Matchers::WithinRel;

TEST_CASE("MfccExtractor initialization", "[mfcc]") {
    AudioConfig audio_config;
    MfccConfig mfcc_config;

    REQUIRE_NOTHROW(MfccExtractor(audio_config, mfcc_config));
}

TEST_CASE("MfccExtractor extracts correct number of coefficients", "[mfcc]") {
    AudioConfig audio_config;
    audio_config.sample_rate = 16000;

    MfccConfig mfcc_config;
    mfcc_config.num_mfcc_coeffs = 13;
    mfcc_config.fft_size = 512;

    MfccExtractor extractor(audio_config, mfcc_config);

    // Create test audio (sine wave at 440Hz)
    std::vector<AudioSample> test_audio(512);
    for (std::size_t i = 0; i < test_audio.size(); ++i) {
        float t = static_cast<float>(i) / 16000.0f;
        test_audio[i] = static_cast<AudioSample>(
            32767.0f * 0.5f * std::sin(2.0f * 3.14159f * 440.0f * t)
        );
    }

    auto features = extractor.extract(AudioSpan{test_audio});

    REQUIRE(features.size() == mfcc_config.num_mfcc_coeffs);
}

TEST_CASE("MfccExtractor batch extraction", "[mfcc]") {
    AudioConfig audio_config;
    audio_config.sample_rate = 16000;
    audio_config.frame_size_ms = 30;
    audio_config.hop_size_ms = 10;

    MfccConfig mfcc_config;

    MfccExtractor extractor(audio_config, mfcc_config);

    // 500ms of audio (sine wave at 300Hz)
    std::vector<AudioSample> test_audio(8000, 0);
    for (std::size_t i = 0; i < test_audio.size(); ++i) {
        float t = static_cast<float>(i) / 16000.0f;
        test_audio[i] = static_cast<AudioSample>(
            32767.0f * 0.3f * std::sin(2.0f * 3.14159f * 300.0f * t)
        );
    }

    auto features = extractor.extract_batch(AudioSpan{test_audio});

    // With 30ms frames and 10ms hop, expect multiple frames
    REQUIRE(features.size() > 1);

    // Each frame should have correct coefficient count
    for (const auto& frame : features) {
        REQUIRE(frame.size() == mfcc_config.num_mfcc_coeffs);
    }
}

TEST_CASE("MfccExtractor handles empty input", "[mfcc]") {
    AudioConfig audio_config;
    MfccConfig mfcc_config;

    MfccExtractor extractor(audio_config, mfcc_config);

    std::vector<AudioSample> empty_audio;
    auto features = extractor.extract_batch(AudioSpan{empty_audio});

    REQUIRE(features.empty());
}

TEST_CASE("MfccExtractor produces finite values", "[mfcc]") {
    AudioConfig audio_config;
    MfccConfig mfcc_config;

    MfccExtractor extractor(audio_config, mfcc_config);

    SECTION("Silence - all zeros") {
        std::vector<AudioSample> silence(512, 0);
        auto features = extractor.extract(AudioSpan{silence});

        for (float coeff : features) {
            REQUIRE(std::isfinite(coeff));
        }
    }

    SECTION("Maximum amplitude - clipping") {
        std::vector<AudioSample> loud(512, 32767);
        auto features = extractor.extract(AudioSpan{loud});

        for (float coeff : features) {
            REQUIRE(std::isfinite(coeff));
        }
    }

    SECTION("Random noise") {
        std::vector<AudioSample> noise(512);
        for (auto& sample : noise) {
            sample = static_cast<AudioSample>(rand() % 65536 - 32768);
        }
        auto features = extractor.extract(AudioSpan{noise});

        for (float coeff : features) {
            REQUIRE(std::isfinite(coeff));
        }
    }
}
