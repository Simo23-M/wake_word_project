# Dependencies management for wake_word_detector

include(FetchContent)
find_package(Threads REQUIRED)

# PortAudio - Audio capture
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PORTAUDIO portaudio-2.0)
endif()

if(NOT PORTAUDIO_FOUND)
    message(STATUS "PortAudio not found via pkg-config, using FetchContent")
    FetchContent_Declare(
        portaudio
        GIT_REPOSITORY https://github.com/PortAudio/portaudio.git
        GIT_TAG v19.7.0
    )
    set(PA_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(PA_BUILD_STATIC ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(portaudio)
    set(PORTAUDIO_LIBRARIES portaudio_static)
    set(PORTAUDIO_INCLUDE_DIRS ${portaudio_SOURCE_DIR}/include)
endif()

# libsndfile - Audio file I/O
if(PkgConfig_FOUND)
    pkg_check_modules(SNDFILE sndfile)
endif()

if(NOT SNDFILE_FOUND)
    message(STATUS "libsndfile not found via pkg-config, using FetchContent")
    FetchContent_Declare(
        libsndfile
        GIT_REPOSITORY https://github.com/libsndfile/libsndfile.git
        GIT_TAG 1.2.2
    )
    set(BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(libsndfile)
    set(SNDFILE_LIBRARIES sndfile)
    set(SNDFILE_INCLUDE_DIRS ${libsndfile_SOURCE_DIR}/include)
endif()

# ONNX Runtime - Inference engine
FetchContent_Declare(
    onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnxruntime)

set(ONNXRUNTIME_INCLUDE_DIRS ${onnxruntime_SOURCE_DIR}/include)
set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/lib)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
    INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_INCLUDE_DIRS}
)

# For NPU support (future)
if(ENABLE_NPU)
    message(STATUS "NPU support enabled - looking for vendor SDK")
    # Rockchip RKNN
    find_path(RKNN_INCLUDE_DIR rknn_api.h
        PATHS /usr/include /usr/local/include
    )
    if(RKNN_INCLUDE_DIR)
        message(STATUS "Found RKNN SDK: ${RKNN_INCLUDE_DIR}")
        add_definitions(-DHAS_RKNN_NPU)
    endif()
endif()

# whisper.cpp - Local speech-to-text for the utterance after the wake word
FetchContent_Declare(
    whisper
    GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp.git
    GIT_TAG v1.8.7
    GIT_SHALLOW TRUE
)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(whisper)
