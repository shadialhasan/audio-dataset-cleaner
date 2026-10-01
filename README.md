# 🎙️ audio-dataset-cleaner

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-brightgreen.svg)](https://python.org)
[![Domain: Speech AI](https://img.shields.io/badge/Domain-Voice%20Cloning%20%7C%20TTS-orange.svg)](https://github.com/MobileConduit)

A production-grade Python CLI utility designed to automate the cleaning, silence trimming, peak normalization, and metadata generation for voice datasets targeting **Coqui XTTS-v2, RVC (Retrieval-based Voice Conversion), and VITS**.

---

## ⚡ Key Features

- **Automated Silence Trimming**: Energy-based thresholding trims leading and trailing dead silence.
- **Peak & RMS Normalization**: Normalizes audio levels to -1 dBFS to prevent clipping and gradient instability during model fine-tuning.
- **Duration Filtering**: Automatically filters out audio fragments that are too short (<1s) or too long (>12s) to stabilize attention matrices.
- **Standardized Metadata Export**: Automatically generates `metadata.csv` compliant with LJSpeech and XTTS formatting.
- **Zero Heavy Dependency Mode**: Built with standard library and NumPy/SciPy compatibility.

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/MobileConduit/audio-dataset-cleaner.git
cd audio-dataset-cleaner
pip install -r requirements.txt
```

### 2. Usage
Place your raw `.wav` recordings in a folder (e.g. `raw_audio/`), then run:

```bash
python audio_cleaner.py --input raw_audio --output dataset_clean --threshold -40 --min-dur 1.2 --max-dur 11.5
```

### CLI Options:
| Flag | Description | Default |
|---|---|---|
| `--input`, `-i` | Input directory of audio files | `raw_audio` |
| `--output`, `-o` | Output directory for normalized dataset | `dataset_clean` |
| `--threshold`, `-t` | Silence detection threshold in dB | `-40.0` |
| `--min-dur` | Minimum clip length in seconds | `1.0` |
| `--max-dur` | Maximum clip length in seconds | `12.0` |

---

## 👤 Author & Maintainer

**Eng. MHD. Shadi AL-Hasan**  
- **Role:** Executive CTO & Enterprise Solutions Architect  
- **Email:** [mhd.shadi.alhasan@gmail.com](mailto:mhd.shadi.alhasan@gmail.com)  
- **Phone / WhatsApp:** [+963934005922](tel:+963934005922)  
- **Location:** Damascus, Syria  
- **GitHub:** [MobileConduit](https://github.com/MobileConduit)  

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  
Copyright (c) 2026 **MHD. Shadi AL-Hasan**. All rights reserved.

