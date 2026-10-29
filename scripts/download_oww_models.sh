#!/usr/bin/env bash
# Download the openWakeWord ONNX models into models/.
# melspectrogram + embedding are shared by every wake word; the rest are classifiers.
# Usage: download_oww_models.sh [wakeword ...]   (default: alexa_v0.1)
set -euo pipefail
VERSION="v0.5.1"
WAKEWORDS=("${@:-alexa_v0.1}")
cd "$(dirname "$0")/../models"
for name in melspectrogram embedding_model "${WAKEWORDS[@]}"; do
  [[ -f "${name}.onnx" ]] && { echo "${name}.onnx already present"; continue; }
  curl -L --fail -o "${name}.onnx" \
    "https://github.com/dscripka/openWakeWord/releases/download/${VERSION}/${name}.onnx"
done
