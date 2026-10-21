#!/usr/bin/env bash
# Download a whisper.cpp ggml model into models/ (default: base, ~140 MB).
# Smaller/faster on ARM: tiny. Better accuracy: small.
set -euo pipefail
MODEL="${1:-base}"
cd "$(dirname "$0")/../models"
curl -L --fail -o "ggml-${MODEL}.bin" \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${MODEL}.bin"
