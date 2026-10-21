/**
 * @file inference_backend.hpp
 * @brief Abstract interface for neural network inference backends.
 *
 * Defines a backend-agnostic interface for running wake word detection
 * models. Concrete implementations can target CPU, GPU, or NPU hardware.
 */

#pragma once

#include "wake_word/config.hpp"
#include "wake_word/types.hpp"

#include <memory>
#include <string_view>

namespace wake_word {

/**
 * @class InferenceBackend
 * @brief Abstract base class for inference engine implementations.
 *
 * Provides a uniform interface for loading models and running inference
 * across different hardware accelerators (CPU, GPU, NPU).
 *
 * To add a new backend:
 * 1. Inherit from InferenceBackend
 * 2. Implement load_model() and infer()
 * 3. Register in create_backend() factory function
 */
class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    /**
     * @brief Load a model from file.
     * @param model_path Path to the ONNX model file.
     * @return Success or error with description.
     */
    [[nodiscard]] virtual Result<void> load_model(std::string_view model_path) = 0;

    /**
     * @brief Run inference on extracted features.
     * @param features MFCC feature matrix (frames x coefficients).
     * @return Model output (typically class probabilities).
     */
    [[nodiscard]] virtual Result<FeatureVector> infer(const FeatureMatrix& features) = 0;

    /**
     * @brief Get human-readable backend name.
     * @return Backend identifier string.
     */
    [[nodiscard]] virtual std::string_view backend_name() const noexcept = 0;

    /**
     * @brief Check if a model is currently loaded.
     * @return True if ready for inference.
     */
    [[nodiscard]] virtual bool is_loaded() const noexcept = 0;
};

/**
 * @brief Create default inference backend based on configuration.
 * @param config Model configuration parameters.
 * @return Unique pointer to backend instance.
 */
[[nodiscard]] std::unique_ptr<InferenceBackend> create_backend(const ModelConfig& config);

/**
 * @brief Available backend types for explicit selection.
 */
enum class BackendType {
    CPU,        ///< ONNX Runtime CPU execution provider
    GPU,        ///< ONNX Runtime CUDA/DirectML provider
    RKNN,       ///< Rockchip NPU (ARM devices)
    OpenVINO,   ///< Intel NPU/GPU acceleration
    Auto        ///< Automatic best-available selection
};

/**
 * @brief Create specific backend type.
 * @param type Desired backend type.
 * @param config Model configuration parameters.
 * @return Unique pointer to backend instance.
 */
[[nodiscard]] std::unique_ptr<InferenceBackend> create_backend(BackendType type, const ModelConfig& config);

} // namespace wake_word
