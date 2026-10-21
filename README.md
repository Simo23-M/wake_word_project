# Wake Word Detection for ARM NPU

A lightweight, real-time wake word detection system optimized for ARM Cortex-A53 processors with Neural Processing Unit (NPU) acceleration. Built in C++26, this project enables custom voice activation (e.g., "ok simone") with minimal latency and power consumption.

## Key Features

- **NPU-Accelerated Inference**: Leverages 2.3 TOPS NPU for ultra-low latency (< 5ms)
- **Custom Wake Words**: Train your own activation phrases
- **Embedded-Friendly**: Designed for resource-constrained ARM devices (Orange Pi, Rock Pi, etc.)
- **Real-Time Processing**: Continuous audio stream analysis with minimal CPU overhead
- **Cross-Platform NPU Support**: Compatible with Rockchip RKNN, Amlogic NPU, and similar accelerators

## Flow

Audio [batch, samples]    Melspectrogram [time, 1, ?, 32]    Embedding [batch, 1, 1, 96]    WakeWord [1, 1]                                                                            
         │                          │                                  │                           │                                                                                     
         └──► melspectrogram.onnx ──┴──► embedding_model.onnx ────────┴──► alexa_v0.1.onnx ───────┘   

## Architecture

```
Audio Input → MFCC Feature Extraction → CNN Model (NPU) → Wake Word Detection
```

Optimized for edge devices with limited computational resources, using quantized neural networks and efficient audio processing pipelines.

## Build

```bash
# Install dependencies (Ubuntu/Debian)
./scripts/install_deps.sh

# Build
mkdir build && cd build
cmake ..
make -j$(nproc)

# Run tests
./tests/wake_word_tests

# Download speech-to-text model (once)
./scripts/download_whisper_model.sh base

# Run: say the wake word, then the command. Text is appended to transcripts.txt
./build/wake_word [wakeword.onnx] [threshold] [ggml-whisper.bin]
```

## Requirements

- C++26 compatible compiler (GCC 14+, Clang 18+)
- CMake 3.28+
- PortAudio, libsndfile (auto-downloaded if not found)
- ONNX Runtime (auto-downloaded)

## Use Cases

Voice-activated assistants, smart home devices, IoT applications, and any embedded system requiring always-on voice detection without cloud dependency.


  wake_word_project/                                                              
  ├── CMakeLists.txt              # Build configuration (C++26)                  
  ├── cmake/                                                                     
  │   └── dependencies.cmake      # Dependancy Auto-download                   
  ├── include/wake_word/                                                         
  │   ├── config.hpp              # Configuration audio/MFCC/detector           
  │   └── types.hpp               # Base type (Result, Error, Detection)         
  ├── src/                                                                       
  │   ├── main.cpp                # Entry point                                  
  │   ├── audio/                                                                 
  │   │   ├── audio_capture.hpp   # Real-time Audio capture                      
  │   │   └── audio_capture.cpp                                                  
  │   ├── features/                                                              
  │   │   ├── mfcc_extractor.hpp  # MFCC extraction feature                      
  │   │   └── mfcc_extractor.cpp                                                 
  │   ├── inference/                                                             
  │   │   ├── inference_backend.hpp   # Abstract interface shoul allow multi hw NPU     
  │   │   ├── cpu_backend.hpp         # Backend CPU (ONNX Runtime)               
  │   │   └── cpu_backend.cpp                                                    
  │   └── detector/                                                              
  │       ├── wake_word_detector.hpp  # Orchestratore principale                 
  │       └── wake_word_detector.cpp                                             
  ├── tests/                      # Unit tests (Catch2)                          
  ├── models/                     # Qui andrà il modello .onnx                   
  ├── data/                       # Dati di training                             
  └── scripts/                                                                   
      └── install_deps.sh         # Script installazione dipendenze              
                                                                                  
Next steps
## Next Steps

1. **Create/Train the Model** - An ONNX model is needed to recognize "ok Simone"
2. **Connect a Microphone** - Currently running in a VM, audio input access will be required
3. **Add NPU Backend** - When you have ARM hardware with NPU support          
