/**
 * @file cpu_backend.cpp
 * @brief Implementation of ONNX Runtime CPU inference backend.
 */

#include "cpu_backend.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace wake_word {

/**
 * @brief Private implementation details for ONNX Runtime.
 *
 * Uses PIMPL pattern to hide ONNX Runtime headers from public API.
 */
struct CpuBackend::Impl {
    std::unique_ptr<Ort::Env> env;           ///< ONNX Runtime environment
    std::unique_ptr<Ort::Session> session;   ///< Inference session
    std::unique_ptr<Ort::MemoryInfo> memory_info; ///< Memory allocator info

    std::vector<const char*> input_names;    ///< Model input tensor names
    std::vector<const char*> output_names;   ///< Model output tensor names
    std::vector<std::int64_t> input_shape;   ///< Expected input dimensions
    std::vector<std::int64_t> output_shape;  ///< Expected output dimensions
};

CpuBackend::CpuBackend(const ModelConfig& config)
    : impl_(std::make_unique<Impl>())
    , config_(config)
{
    // Initialize ONNX Runtime environment
    impl_->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "wake_word");
    impl_->memory_info = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
    );
}

CpuBackend::~CpuBackend() = default;

CpuBackend::CpuBackend(CpuBackend&&) noexcept = default;
CpuBackend& CpuBackend::operator=(CpuBackend&&) noexcept = default;

Result<void> CpuBackend::load_model(std::string_view model_path) {
    try {
        // Configure session with optimization settings
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(config_.num_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        std::string path_str(model_path);
        impl_->session = std::make_unique<Ort::Session>(
            *impl_->env,
            path_str.c_str(),
            session_options
        );

        // Extract input tensor metadata
        Ort::AllocatorWithDefaultOptions allocator;

        auto input_name = impl_->session->GetInputNameAllocated(0, allocator);
        impl_->input_names.clear();
        impl_->input_names.push_back(input_name.get());

        auto input_info = impl_->session->GetInputTypeInfo(0);
        auto input_tensor_info = input_info.GetTensorTypeAndShapeInfo();
        impl_->input_shape = input_tensor_info.GetShape();

        // Extract output tensor metadata
        auto output_name = impl_->session->GetOutputNameAllocated(0, allocator);
        impl_->output_names.clear();
        impl_->output_names.push_back(output_name.get());

        auto output_info = impl_->session->GetOutputTypeInfo(0);
        auto output_tensor_info = output_info.GetTensorTypeAndShapeInfo();
        impl_->output_shape = output_tensor_info.GetShape();

        loaded_ = true;
        return {};

    } catch (const Ort::Exception& e) {
        return std::unexpected(Error::make(
            ErrorCode::ModelLoadError,
            std::string("ONNX Runtime error: ") + e.what()
        ));
    } catch (const std::exception& e) {
        return std::unexpected(Error::make(
            ErrorCode::ModelLoadError,
            e.what()
        ));
    }
}

Result<FeatureVector> CpuBackend::infer(const FeatureMatrix& features) {
    if (!loaded_) {
        return std::unexpected(Error::make(
            ErrorCode::InferenceError,
            "Model not loaded"
        ));
    }

    try {
        // Flatten feature matrix into contiguous buffer
        std::vector<float> input_data;
        for (const auto& frame : features) {
            input_data.insert(input_data.end(), frame.begin(), frame.end());
        }

        // Handle dynamic dimensions (-1 in shape)
        auto shape = impl_->input_shape;
        if (!shape.empty() && shape[0] == -1) {
            shape[0] = 1; // Batch size = 1
        }
        if (shape.size() > 1 && shape[1] == -1) {
            shape[1] = static_cast<std::int64_t>(features.size()); // Time steps
        }

        // Calculate total elements and resize input
        const auto total_elements = std::accumulate(
            shape.begin(), shape.end(),
            std::int64_t{1},
            std::multiplies<>{}
        );
        input_data.resize(static_cast<std::size_t>(total_elements), 0.0f);

        // Create input tensor
        auto input_tensor = Ort::Value::CreateTensor<float>(
            *impl_->memory_info,
            input_data.data(),
            input_data.size(),
            shape.data(),
            shape.size()
        );

        // Execute inference
        auto output_tensors = impl_->session->Run(
            Ort::RunOptions{nullptr},
            impl_->input_names.data(),
            &input_tensor,
            1,
            impl_->output_names.data(),
            1
        );

        // Extract output values
        const float* output_data = output_tensors[0].GetTensorData<float>();
        auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        const auto output_size = output_info.GetElementCount();

        return FeatureVector(output_data, output_data + output_size);

    } catch (const Ort::Exception& e) {
        return std::unexpected(Error::make(
            ErrorCode::InferenceError,
            std::string("ONNX Runtime inference error: ") + e.what()
        ));
    } catch (const std::exception& e) {
        return std::unexpected(Error::make(
            ErrorCode::InferenceError,
            e.what()
        ));
    }
}

std::string_view CpuBackend::backend_name() const noexcept {
    return "CPU (ONNX Runtime)";
}

bool CpuBackend::is_loaded() const noexcept {
    return loaded_;
}

// Factory implementations

std::unique_ptr<InferenceBackend> create_backend(const ModelConfig& config) {
    return std::make_unique<CpuBackend>(config);
}

std::unique_ptr<InferenceBackend> create_backend(BackendType type, const ModelConfig& config) {
    switch (type) {
        case BackendType::CPU:
        case BackendType::Auto:
        default:
            return std::make_unique<CpuBackend>(config);
        // Future: Add GPU, RKNN, OpenVINO backends
    }
}

} // namespace wake_word
