# Wake Word Detection for ARM NPU

A lightweight, real-time wake word detection system optimized for ARM Cortex-A53 processors with Neural Processing Unit (NPU) acceleration. Built in C++26, this project enables custom voice activation (e.g., "ok simone") with minimal latency and power consumption.

## Key Features

- **NPU-Accelerated Inference**: Leverages 2.3 TOPS NPU for ultra-low latency (< 5ms)
- **Custom Wake Words**: Train your own activation phrases
- **Embedded-Friendly**: Designed for resource-constrained ARM devices (Orange Pi, Rock Pi, etc.)
- **Real-Time Processing**: Continuous audio stream analysis with minimal CPU overhead
- **Cross-Platform NPU Support**: Compatible with Rockchip RKNN, Amlogic NPU, and similar accelerators

## Architecture

```
Audio Input → MFCC Feature Extraction → CNN Model (NPU) → Wake Word Detection
```

Optimized for edge devices with limited computational resources, using quantized neural networks and efficient audio processing pipelines.

## HW requirements

Multiplatform

## Use Cases

Voice-activated assistants, smart home devices, IoT applications, and any embedded system requiring always-on voice detection without cloud dependency.
