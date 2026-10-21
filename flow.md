```
┌─────────────┐
│  Microfono  │
└──────┬──────┘
       │ Audio 16kHz
       ↓
┌─────────────────┐
│ MFCC Extract    │ ← CPU (2ms)
│ (Aquila/libROSA)│
└──────┬──────────┘
       │ Features 13x100
       ↓
┌─────────────────┐
│ Wake Word CNN   │ ← NPU (2-5ms) ⚡
│ "ok simone"?    │
└──────┬──────────┘
       │ Confidence > 0.85?
       ↓
   [SI] Trigger
       │
       ↓
┌─────────────────┐
│ Registra 3 sec  │ ← CPU/DMA
│ buffer audio    │
└──────┬──────────┘
       │
       ↓ Invia a cloud o processa con CPU
┌─────────────────┐
│ Speech-to-Text  │ ← API esterna
│ (Whisper API)   │    oppure CPU locale
└─────────────────┘
```
