/**
 * @file cpu_backend.hpp
 * @brief CPU inference backend using ONNX Runtime.
 *
 * Provides neural network inference on CPU using ONNX Runtime.
 * Supports multi-threaded execution and various optimization levels.
 */

#pragma once

#include "inference_backend.hpp"

#include <memory>
#include <vector>

namespace Ort {
    struct Env;
    struct Session;
    struct SessionOptions;
    struct MemoryInfo;
}

namespace wake_word {

/**
 * @class CpuBackend
 * @brief ONNX Runtime CPU inference implementation.
 *
 * Uses ONNX Runtime's CPU execution provider for portable inference.
 * Supports dynamic batch sizes and automatic input shape handling.
 *
 * Features:
 * - Multi-threaded inference (configurable thread count)
 * - Graph optimization (constant folding, operator fusion)
 * - INT8 quantized model support for faster inference
 */
class CpuBackend : public InferenceBackend {
public:
    /**
     * @brief Construct CPU backend with given configuration.
     * @param config Model configuration (threads, quantization).
     */
    explicit CpuBackend(const ModelConfig& config);

    /**
     * @brief Destructor. Releases ONNX Runtime resources.
     */
    ~CpuBackend() override;

    CpuBackend(const CpuBackend&) = delete;
    CpuBackend& operator=(const CpuBackend&) = delete;

    /**
     * @brief Move constructor.
     * @param other Source instance.
     */
    CpuBackend(CpuBackend&&) noexcept;

    /**
     * @brief Move assignment operator.
     * @param other Source instance.
     * @return Reference to this.
     */
    CpuBackend& operator=(CpuBackend&&) noexcept;

    /**
     * @brief Load ONNX model from file.
     * @param model_path Path to .onnx model file.
     * @return Success or error with ONNX Runtime message.
     */
    [[nodiscard]] Result<void> load_model(std::string_view model_path) override;

    /**
     * @brief Run inference on MFCC features.
     * @param features Input feature matrix.
     * @return Output probabilities or error.
     *
     * Input is automatically reshaped to match model expectations.
     * For wake word detection, output[0] is typically the wake word confidence.
     */
    [[nodiscard]] Result<FeatureVector> infer(const FeatureMatrix& features) override;

    /**
     * @brief Get backend identifier.
     * @return "CPU (ONNX Runtime)"
     */
    [[nodiscard]] std::string_view backend_name() const noexcept override;

    /**
     * @brief Check if model is loaded and ready.
     * @return True if inference can be performed.
     */
    [[nodiscard]] bool is_loaded() const noexcept override;

private:
    /// PIMPL for ONNX Runtime internals
    struct Impl;
    std::unique_ptr<Impl> impl_;
    ModelConfig config_;  ///< Model configuration
    bool loaded_ = false; ///< Model load state
};

} // namespace wake_word
