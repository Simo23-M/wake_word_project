/**
 * @file test_audio.cpp
 * @brief Unit tests for configuration and type definitions.
 */

#include <catch2/catch_test_macros.hpp>

#include "wake_word/config.hpp"
#include "wake_word/types.hpp"

using namespace wake_word;

TEST_CASE("AudioConfig calculations", "[config]") {
    AudioConfig config;
    config.sample_rate = 16000;
    config.frame_size_ms = 30;
    config.hop_size_ms = 10;

    SECTION("samples_per_frame computes correctly") {
        // 16000 Hz * 30 ms / 1000 = 480 samples
        REQUIRE(config.samples_per_frame() == 480);
    }

    SECTION("hop_samples computes correctly") {
        // 16000 Hz * 10 ms / 1000 = 160 samples
        REQUIRE(config.hop_samples() == 160);
    }
}

TEST_CASE("Error creation", "[types]") {
    auto error = Error::make(ErrorCode::AudioDeviceNotFound, "No device");

    REQUIRE(error.code == ErrorCode::AudioDeviceNotFound);
    REQUIRE(error.message == "No device");
}

TEST_CASE("Result type behavior", "[types]") {
    SECTION("Success case holds value") {
        Result<int> result = 42;
        REQUIRE(result.has_value());
        REQUIRE(result.value() == 42);
    }

    SECTION("Error case holds error") {
        Result<int> result = std::unexpected(
            Error::make(ErrorCode::InvalidConfig, "Bad config")
        );
        REQUIRE(!result.has_value());
        REQUIRE(result.error().code == ErrorCode::InvalidConfig);
    }
}

TEST_CASE("Detection struct", "[types]") {
    Detection d{};
    d.confidence = 0.95f;
    d.timestamp_ms = 1000;
    d.is_detected = true;

    REQUIRE(d.confidence == 0.95f);
    REQUIRE(d.timestamp_ms == 1000);
    REQUIRE(static_cast<bool>(d) == true);

    Detection d2{};
    d2.is_detected = false;
    REQUIRE(static_cast<bool>(d2) == false);
}
