/**
 * @file test_recorder.cpp
 * @brief Endpointing tests for UtteranceRecorder.
 */

#include <catch2/catch_test_macros.hpp>

#include "detector/utterance_recorder.hpp"

using namespace wake_word;

namespace {
    const AudioBuffer kSpeech(1280, 3000);  // 80 ms loud
    const AudioBuffer kSilence(1280, 0);    // 80 ms quiet

    int chunks_until_done(UtteranceRecorder& r, int speech_chunks) {
        int n = 0;
        for (; n < speech_chunks; ++n) if (r.push(kSpeech)) return n + 1;
        while (!r.push(kSilence)) ++n;
        return n + 1;
    }
}

TEST_CASE("UtteranceRecorder endpointing", "[recorder]") {
    UtteranceRecorder r;

    SECTION("ends after 1 s of silence following speech") {
        // 10 speech chunks + 13 silent chunks (13 * 80 = 1040 ms)
        REQUIRE(chunks_until_done(r, 10) == 23);
        REQUIRE(r.has_speech());
        REQUIRE(r.audio.size() == 23 * 1280);
    }

    SECTION("gives up when nobody speaks") {
        REQUIRE(chunks_until_done(r, 0) == 38);  // 38 * 80 = 3040 ms
        REQUIRE_FALSE(r.has_speech());
    }

    SECTION("caps at max duration") {
        REQUIRE(chunks_until_done(r, 1000) == 125);  // 125 * 80 = 10 s
    }

    SECTION("reset clears state") {
        chunks_until_done(r, 5);
        r.reset();
        REQUIRE(r.audio.empty());
        REQUIRE_FALSE(r.has_speech());
    }
}
