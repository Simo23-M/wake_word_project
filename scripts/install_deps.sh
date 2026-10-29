#!/bin/bash
# Install dependencies for wake_word_detector

set -e

echo "=== Wake Word Detector - Dependency Installation ==="
echo ""

# Detect package manager
if command -v apt-get &> /dev/null; then
    PKG_MANAGER="apt"
elif command -v dnf &> /dev/null; then
    PKG_MANAGER="dnf"
elif command -v pacman &> /dev/null; then
    PKG_MANAGER="pacman"
else
    echo "Error: Unsupported package manager"
    exit 1
fi

echo "Detected package manager: $PKG_MANAGER"
echo ""

# Install system dependencies
echo "Installing system dependencies..."

case $PKG_MANAGER in
    apt)
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            cmake \
            pkg-config \
            libasound2-dev \
            libportaudio2 \
            portaudio19-dev \
            libsndfile1-dev \
            git
        ;;
    dnf)
        sudo dnf install -y \
            gcc-c++ \
            cmake \
            pkg-config \
            alsa-lib-devel \
            portaudio-devel \
            libsndfile-devel \
            git
        ;;
    pacman)
        sudo pacman -S --needed \
            base-devel \
            cmake \
            pkg-config \
            alsa-lib \
            portaudio \
            libsndfile \
            git
        ;;
esac

echo ""
echo "=== Dependencies installed successfully ==="
echo ""
echo "Next steps:"
echo "  1. ./scripts/download_oww_models.sh && ./scripts/download_whisper_model.sh tiny"
echo "  2. cmake -B build -DCMAKE_BUILD_TYPE=Release"
echo "  3. cmake --build build -j2"
echo ""
echo "Note: ONNX Runtime will be downloaded automatically during cmake."
