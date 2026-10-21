# Wake Word Detection - Guida allo Studio

Questa guida ti accompagna nella comprensione teorica e pratica del sistema di wake word detection.

---

## Indice

1. [Panoramica del Sistema](#1-panoramica-del-sistema)
2. [Teoria del Riconoscimento Vocale](#2-teoria-del-riconoscimento-vocale)
3. [Pipeline Audio](#3-pipeline-audio)
4. [Feature Extraction](#4-feature-extraction)
5. [Neural Network e Inference](#5-neural-network-e-inference)
6. [OpenWakeWord Architecture](#6-openwakeword-architecture)
7. [Scelte Architetturali C++](#7-scelte-architetturali-c)
8. [Risorse per Approfondire](#8-risorse-per-approfondire)

---

## 1. Panoramica del Sistema

### Cosa fa un Wake Word Detector?

Un wake word detector è un sistema **always-on** che ascolta continuamente l'audio ambientale cercando una specifica parola o frase di attivazione (es. "Alexa", "Hey Siri", "Ok Google").

**Requisiti chiave:**
- **Bassa latenza**: risposta in millisecondi
- **Basso consumo**: deve girare 24/7
- **Alta accuratezza**: minimizzare falsi positivi/negativi
- **Real-time**: elaborazione streaming, non batch

### Architettura del nostro sistema

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Microfono  │───►│   Audio     │───►│  Feature    │───►│   Neural    │───► Detection
│             │    │  Capture    │    │ Extraction  │    │  Network    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                         │                  │                   │
                    PortAudio          Melspectrogram      ONNX Runtime
                    16kHz mono         + Embedding          3 modelli
```

**File correlati:**
- `src/audio/audio_capture.cpp` - Cattura audio
- `src/inference/openwakeword_backend.cpp` - Pipeline inference
- `src/detector/wake_word_detector.cpp` - Orchestrazione

---

## 2. Teoria del Riconoscimento Vocale

### Il suono come segnale

Il suono è un'onda di pressione. Quando digitalizziamo:

```
Onda sonora → Microfono → Tensione elettrica → ADC → Numeri discreti
```

**Parametri di campionamento:**
- **Sample rate**: 16000 Hz (16000 numeri al secondo)
- **Bit depth**: 16 bit (valori da -32768 a +32767)
- **Canali**: 1 (mono)

**Perché 16kHz?**
Il teorema di Nyquist dice che per catturare frequenze fino a F Hz, serve campionare a 2F Hz. La voce umana ha informazioni utili fino a ~8kHz, quindi 16kHz è sufficiente.

```cpp
// Da config.hpp
struct AudioConfig {
    std::uint32_t sample_rate = 16000;  // Nyquist per speech
    std::uint16_t channels = 1;          // Mono basta per wake word
    std::uint16_t bits_per_sample = 16;  // Standard PCM
};
```

### Framing e Windowing

L'audio non viene analizzato campione per campione, ma in **frame** (finestre temporali):

```
Audio continuo: ──────────────────────────────────────────►

Frame 1:        [████████]
Frame 2:            [████████]        ← Overlap (hop)
Frame 3:                [████████]
```

**Parametri tipici:**
- Frame size: 25-30ms (400-480 samples @ 16kHz)
- Hop size: 10ms (160 samples) - overlap del 60-70%

**Perché overlap?** Evita di "tagliare" suoni a metà frame.

### Dominio del tempo vs frequenza

Il segnale audio grezzo è nel **dominio del tempo** (ampiezza vs tempo). Per il riconoscimento vocale, lavoriamo nel **dominio della frequenza**:

```
Dominio tempo:  "come suona nel tempo"
Dominio freq:   "quali frequenze contiene"
```

La trasformata di Fourier (FFT) converte tra i due:

```
Segnale temporale ──[FFT]──► Spettro (magnitudine per frequenza)
```

---

## 3. Pipeline Audio

### PortAudio

PortAudio è una libreria cross-platform per audio I/O. Gestisce:
- Enumerazione dispositivi
- Apertura stream
- Callback quando arrivano dati

```cpp
// Da audio_capture.cpp
int AudioCapture::pa_callback(
    const void* input,          // Buffer audio dal microfono
    void* output,               // Non usato (solo input)
    unsigned long frame_count,  // Quanti campioni
    ...
    void* user_data             // Puntatore al nostro oggetto
) {
    auto* capture = static_cast<AudioCapture*>(user_data);
    // Passa i dati al callback utente
    capture->callback_(AudioSpan{samples, frame_count});
    return paContinue;
}
```

**Concetti chiave:**
- **Callback-based**: PortAudio chiama la nostra funzione quando ha dati
- **Real-time thread**: il callback gira in un thread ad alta priorità
- **Non bloccare mai**: nel callback, niente allocazioni o I/O

### Ring Buffer

Per disaccoppiare il thread audio (real-time) dal thread di elaborazione:

```
Thread Audio ──► [Ring Buffer] ──► Thread Detection
  (veloce)         (FIFO)           (può essere lento)
```

```cpp
// Da wake_word_detector.cpp
void WakeWordDetector::on_audio(AudioSpan audio) {
    std::lock_guard lock(buffer_mutex_);
    for (auto sample : audio) {
        audio_buffer_[buffer_pos_++] = sample;
    }
}
```

---

## 4. Feature Extraction

### Perché non usare l'audio grezzo?

L'audio grezzo (waveform) contiene troppe informazioni ridondanti:
- Volume assoluto (irrilevante)
- Fase del segnale (irrilevante per speech)
- Dettagli non percepibili dall'orecchio

Le **features** estraggono solo l'informazione utile.

### MFCC (Mel-Frequency Cepstral Coefficients)

Pipeline classica (implementata in `mfcc_extractor.cpp`):

```
Audio → Pre-emphasis → Windowing → FFT → Mel Filterbank → Log → DCT → MFCC
```

**Passo 1: Pre-emphasis**
```cpp
// Amplifica le alte frequenze (la voce ha più energia nelle basse)
y[n] = x[n] - 0.97 * x[n-1]
```

**Passo 2: Windowing (Hamming)**
```cpp
// Smussa i bordi del frame per evitare artefatti FFT
window[n] = 0.54 - 0.46 * cos(2π * n / (N-1))
```

**Passo 3: FFT**
```
Segnale [512 samples] → FFT → Spettro [257 bins]
```

**Passo 4: Mel Filterbank**

La scala Mel approssima la percezione umana delle frequenze:
- Basse frequenze: alta risoluzione
- Alte frequenze: bassa risoluzione

```cpp
// Conversione Hz → Mel
float hz_to_mel(float hz) {
    return 2595.0f * log10(1.0f + hz / 700.0f);
}
```

I filtri triangolari sommano le energie in bande percettivamente uniformi:

```
Frequenza (Hz): 0    500   1000  2000  4000  8000
                ▲     ▲     ▲     ▲     ▲     ▲
Mel scale:      |█|█|█|█|█|█| █ | █ |  █  |   █   |
                ← denso →            ← sparso →
```

**Passo 5: Log compression**
```cpp
mel_energy[i] = log(max(energy, 1e-10f));
```
Comprime il range dinamico (orecchio percepisce logaritmicamente).

**Passo 6: DCT (Discrete Cosine Transform)**

Decorrelazione delle features. I primi 13 coefficienti catturano la "forma" dello spettro.

### Melspectrogram (OpenWakeWord)

OpenWakeWord usa una variante semplificata:

```
Audio → Mel Filterbank → Log → (niente DCT)
```

Output: matrice [time × 32 mel bins]

Il modello neurale impara la trasformazione ottimale invece di usare DCT fisso.

---

## 5. Neural Network e Inference

### Perché reti neurali?

I classificatori tradizionali (GMM, HMM) richiedono feature engineering manuale. Le reti neurali:
- Imparano features automaticamente
- Gestiscono variabilità (accenti, rumore)
- Scalano con più dati

### Architettura tipica per Wake Word

```
Input [time × features]
        │
   ┌────▼────┐
   │  Conv2D │  ← Estrae pattern locali
   └────┬────┘
        │
   ┌────▼────┐
   │  Conv2D │  ← Pattern più complessi
   └────┬────┘
        │
   ┌────▼────┐
   │ Flatten │
   └────┬────┘
        │
   ┌────▼────┐
   │  Dense  │  ← Classificazione
   └────┬────┘
        │
   ┌────▼────┐
   │ Sigmoid │  ← Probabilità [0,1]
   └────┴────┘
```

### ONNX Runtime

ONNX (Open Neural Network Exchange) è un formato standard per modelli ML.

**Vantaggi:**
- Framework-agnostico (PyTorch, TensorFlow → ONNX)
- Ottimizzato per inference
- Supporta CPU, GPU, NPU

```cpp
// Da cpu_backend.cpp
auto outputs = session->Run(
    Ort::RunOptions{nullptr},
    input_names.data(),
    &input_tensor,
    1,
    output_names.data(),
    1
);
```

### Quantizzazione

I modelli "full precision" usano float32. La quantizzazione riduce a int8:
- 4x meno memoria
- 2-4x più veloce
- Minima perdita di accuratezza

Critico per dispositivi embedded/NPU.

---

## 6. OpenWakeWord Architecture

### Pipeline a 3 modelli

OpenWakeWord separa la pipeline in componenti riutilizzabili:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Melspectrogram  │────►│    Embedding     │────►│    Wake Word     │
│     Model        │     │     Model        │     │     Model        │
└──────────────────┘     └──────────────────┘     └──────────────────┘
   melspectrogram.onnx     embedding_model.onnx     alexa_v0.1.onnx

   Audio → Mel features    Mel → 96-dim vector     16 embeddings → score
   [1, 1280] → [?, 32]     [76, 32] → [96]         [16, 96] → [1]
```

**Perché separare?**
1. **Melspectrogram**: fisso, non richiede training
2. **Embedding**: condiviso tra tutti i wake word
3. **Wake Word**: piccolo, veloce da addestrare per nuove parole

### Embedding Buffer

Il classificatore guarda gli ultimi 16 frame (~1.3 secondi):

```cpp
// Da openwakeword_backend.cpp
std::deque<std::vector<float>> embedding_buffer_;  // [16 × 96]

// Ad ogni chunk audio:
embedding_buffer_.pop_front();         // Rimuovi il più vecchio
embedding_buffer_.push_back(new_emb);  // Aggiungi il nuovo
```

Questo crea una "finestra scorrevole" che cattura il contesto temporale.

### Training Custom Wake Word

Per creare "Ok Simone":
1. Genera audio sintetico con TTS (Text-to-Speech)
2. Crea esempi negativi (rumore, altre parole)
3. Addestra solo il classificatore finale
4. Esporta in ONNX

Il notebook ufficiale automatizza tutto: ~1 ora su Google Colab.

---

## 7. Scelte Architetturali C++

### C++26 Features Utilizzate

**std::expected (C++23)**
```cpp
// Error handling senza eccezioni
Result<void> load_model(std::string_view path);

// Uso:
if (auto result = load_model("model.onnx"); !result) {
    return std::unexpected(result.error());
}
```

**std::span (C++20)**
```cpp
// View non-owning su array
using AudioSpan = std::span<const AudioSample>;

void process(AudioSpan audio);  // Non copia, solo puntatore + size
```

**std::jthread (C++20)**
```cpp
// Thread con stop automatico
std::jthread detection_thread_([this](std::stop_token stoken) {
    while (!stoken.stop_requested()) {
        // ...
    }
});
// Distruttore fa join automaticamente
```

**std::ranges (C++20)**
```cpp
std::ranges::fill(buffer, 0.0f);
```

### Pattern Utilizzati

**RAII (Resource Acquisition Is Initialization)**
```cpp
class AudioCapture {
    PaStream* stream_ = nullptr;
public:
    ~AudioCapture() {
        stop();  // Cleanup automatico
    }
};
```

**PIMPL (Pointer to Implementation)**
```cpp
// Header pubblico (no dipendenze ONNX)
class CpuBackend {
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Source (dipendenze nascoste)
struct CpuBackend::Impl {
    Ort::Session session;  // ONNX Runtime
};
```

**Strategy Pattern (Backend intercambiabili)**
```cpp
class InferenceBackend {
    virtual Result<FeatureVector> infer(...) = 0;
};

class CpuBackend : public InferenceBackend { ... };
class OpenWakeWordBackend : public InferenceBackend { ... };
// Futuro: RknnBackend, OpenVinoBackend
```

### Thread Safety

```cpp
std::mutex buffer_mutex_;
std::atomic<bool> running_{false};
std::atomic<float> threshold_;

// Lock solo dove serve
void on_audio(AudioSpan audio) {
    std::lock_guard lock(buffer_mutex_);  // RAII lock
    // ... modifica buffer ...
}
```

---

## 8. Risorse per Approfondire

### Teoria Audio/DSP

1. **"Speech and Language Processing"** - Jurafsky & Martin
   - Capitoli su speech recognition
   - Gratuito online: https://web.stanford.edu/~jurafsky/slp3/

2. **"The Scientist and Engineer's Guide to DSP"**
   - FFT, filtering, spectral analysis
   - Gratuito: http://www.dspguide.com/

### Machine Learning per Audio

3. **"Deep Learning for Audio"** - Valerio Velardo (YouTube)
   - Serie completa su audio ML
   - https://www.youtube.com/c/ValerioVelwordo

4. **Librosa Documentation**
   - Implementazione Python di MFCC, Mel, etc.
   - https://librosa.org/doc/

### Wake Word Specifico

5. **OpenWakeWord Paper & Docs**
   - https://github.com/dscripka/openWakeWord
   - Notebook di training

6. **Picovoice Blog**
   - Articoli tecnici su wake word
   - https://picovoice.ai/blog/

### C++ Moderno

7. **"A Tour of C++"** - Bjarne Stroustrup
   - Overview C++20/23

8. **CppReference**
   - Documentazione standard
   - https://en.cppreference.com/

### ONNX Runtime

9. **ONNX Runtime Docs**
   - https://onnxruntime.ai/docs/
   - Esempi C++

---

## Esercizi Suggeriti

### Livello Base
1. Modifica la threshold e osserva l'effetto sui falsi positivi
2. Aggiungi logging del confidence score ad ogni chunk
3. Implementa un contatore di detection

### Livello Intermedio
4. Aggiungi supporto per selezionare il dispositivo audio
5. Implementa salvataggio audio su file quando rileva wake word
6. Crea un test che usa file audio invece del microfono

### Livello Avanzato
7. Implementa un nuovo backend (es. TensorFlow Lite)
8. Addestra un modello custom per "Ok Simone"
9. Ottimizza la pipeline per ridurre la latenza
10. Aggiungi supporto NPU (RKNN o OpenVINO)

---

## Glossario

| Termine | Significato |
|---------|-------------|
| **ADC** | Analog-to-Digital Converter |
| **DCT** | Discrete Cosine Transform |
| **Embedding** | Rappresentazione vettoriale densa |
| **FFT** | Fast Fourier Transform |
| **Frame** | Finestra temporale di audio |
| **Hop** | Spostamento tra frame consecutivi |
| **Mel scale** | Scala di frequenza percettiva |
| **MFCC** | Mel-Frequency Cepstral Coefficients |
| **ONNX** | Open Neural Network Exchange |
| **Quantization** | Riduzione precisione numerica |
| **Sample rate** | Frequenza di campionamento |
| **Spectrogram** | Rappresentazione tempo-frequenza |
| **VAD** | Voice Activity Detection |
