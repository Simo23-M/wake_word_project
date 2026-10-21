/**
 * @file main.cpp
 * @brief Entry point for wake word detection application.
 *
 * Uses OpenWakeWord-compatible models for wake word detection.
 * Default model is "alexa" but any OpenWakeWord model can be used.
 */

#include "detector/wake_word_detector.hpp"
#include "stt/transcriber.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <print>

namespace {
    /// Global flag for graceful shutdown
    std::atomic<bool> g_running{true};

    void signal_handler(int /*signal*/) {
        g_running.store(false);
    }

    /// Deliver the cleaned-up command text.
    /// ponytail: appends to a text file until the real service/API exists; replace the body with the HTTP call.
    void send_payload(const std::string& text) {
        std::ofstream out("transcripts.txt", std::ios::app);
        out << std::format("{:%F %T}\t{}\n",
            std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()), text);
    }
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Default to alexa model, or use command line argument
    std::string model_path = "models/alexa_v0.1.onnx";
    float threshold = 0.5f;
    std::string whisper_model_path = "models/ggml-base.bin";

    if (argc > 1) {
        model_path = argv[1];
    }
    if (argc > 2) {
        threshold = std::stof(argv[2]);
    }
    if (argc > 3) {
        whisper_model_path = argv[3];
    }

    std::println("===========================================");
    std::println("  Wake Word Detector (OpenWakeWord)");
    std::println("===========================================");
    std::println("Model: {}", model_path);
    std::println("Threshold: {:.2f}", threshold);
    std::println("Whisper model: {}", whisper_model_path);
    std::println("");

    // List audio devices
    auto devices = wake_word::AudioCapture::list_devices();
    if (devices && !devices->empty()) {
        std::println("Audio input devices:");
        for (std::size_t i = 0; i < devices->size(); ++i) {
            std::println("  [{}] {}", i, (*devices)[i]);
        }
        std::println("");
    }

    // Create detector
    wake_word::WakeWordDetector detector(model_path, threshold);

    // Initialize
    std::println("Loading models...");
    if (auto result = detector.initialize(); !result) {
        std::println(stderr, "ERROR: {}", result.error().message);
        return 1;
    }

    wake_word::Transcriber transcriber(whisper_model_path);
    if (!transcriber.is_loaded()) {
        std::println(stderr, "ERROR: cannot load whisper model {} (run scripts/download_whisper_model.sh)", whisper_model_path);
        return 1;
    }
    std::println("Models loaded successfully.");
    std::println("");

    // Set detection callback
    detector.set_detection_callback([](const wake_word::Detection& detection) {
        std::println("");
        std::println("***** WAKE WORD DETECTED! *****");
        std::println("  Confidence: {:.1f}%", detection.confidence * 100.0f);
        std::println("Recording...");
    });

    detector.set_utterance_callback([&transcriber](wake_word::AudioSpan audio) {
        auto text = transcriber.transcribe(audio);
        if (!text) {
            std::println(stderr, "ERROR: {}", text.error().message);
            return;
        }
        std::println("Heard: \"{}\"", *text);
        if (!text->empty()) {
            send_payload(*text);
        }
        std::println("Listening for wake word...");
    });

    // Start detection
    if (auto result = detector.start(); !result) {
        std::println(stderr, "ERROR: {}", result.error().message);
        return 1;
    }

    std::println("Listening for wake word... (Ctrl+C to stop)");
    std::println("");

    // Main loop
    while (g_running.load() && detector.is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::println("\nShutting down...");
    detector.stop();

    return 0;
}
