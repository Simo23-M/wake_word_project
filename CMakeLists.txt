cmake_minimum_required(VERSION 3.28)
project(wake_word_detector
    VERSION 0.1.0
    DESCRIPTION "Custom wake word detection with NPU support"
    LANGUAGES C CXX
)

# C++26 standard
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Options
option(ENABLE_NPU "Enable NPU backend support" OFF)
option(ENABLE_TESTS "Build tests" ON)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

# Include custom CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(dependencies)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Source files
set(WAKE_WORD_SOURCES
    src/audio/audio_capture.cpp
    src/features/mfcc_extractor.cpp
    src/inference/cpu_backend.cpp
    src/inference/openwakeword_backend.cpp
    src/detector/wake_word_detector.cpp
    src/stt/transcriber.cpp
)

# Main library
add_library(wake_word_lib STATIC ${WAKE_WORD_SOURCES})
target_link_libraries(wake_word_lib PUBLIC
    ${PORTAUDIO_LIBRARIES}
    ${SNDFILE_LIBRARIES}
    onnxruntime
    whisper
    Threads::Threads
    m
)

target_include_directories(wake_word_lib PUBLIC
    ${PORTAUDIO_INCLUDE_DIRS}
    ${SNDFILE_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# Main executable
add_executable(wake_word src/main.cpp)
target_link_libraries(wake_word PRIVATE wake_word_lib)

# Microphone / wake word diagnostic
add_executable(mic_check src/tools/mic_check.cpp)
target_link_libraries(mic_check PRIVATE wake_word_lib)

# Tests
if(ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Install
install(TARGETS wake_word RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Wake Word Detector Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "NPU support: ${ENABLE_NPU}")
message(STATUS "Tests: ${ENABLE_TESTS}")
message(STATUS "=========================================")
message(STATUS "")
