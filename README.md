# 🎧 Multi-Channel Audio Dataset Generator

This repository provides a tool to generate multi-channel audio datasets using simulated room acoustics and microphone arrays. It is designed for research in multi channel audio, audio enhancement, speech separation, and spatial audio modeling. The tool simulates reverberant environments with configurable source and microphone positions using [Pyroomacoustics](https://github.com/LCAV/pyroomacoustics).

---

## 📌 Features

- Generate multi-speaker room simulations with:
  - Configurable room dimensions
  - Custom reverberation time (RT60)
  - Adjustable Signal-to-Noise Ratio (SNR)
- Support for:
  - Multi-channel microphone arrays (e.g., 4-mic glasses-mounted setup)
  - Target and interfering speakers
- Automatic metadata generation (JSON)
- Audio saved per channel and per simulation

---

## 📁 Directory Structure
├── dataset_generator.py # Main script to generate dataset

├── .... .py             # Script to support generate dataset

├── audio/               # Input audio files (target & interference)
  ├── s1/                # Speaker 1 audio dir
    ├── bbaf2n.wav       # audio file, .wav format
    ├── bbaf3s.wav       # audio file, .wav format
    .
    .
    .
  ├── s2/
  .
  .
  .
  ├── s32/


---
## 🔧 Requirements
pip install -r requirements.txt


---
## 🧪 Example Usage
Edit dataset_generator.py:
  - A_DATA_FOLDER --> audiodir path
  - snr_values --> SNR: list of SNR values (dB)
  - simulation --> simulation type: 'room' or 'free_field'
  - noise --> noise type: 'interfere' or 'babble'
  - num_pairs_per_spk --> Number of sample combination per target speaker with other interfering speaker: int
  - save --> save multi channel audio in output_dir: Boolen
  - verbose --> show progress, prints, plot a 2d room simulation, plot signals at microphones
  - Room size: list of [x,y,z] m^3 configurations
  - RT60: list of t60 values
  - Number of interfering speakers

Run: python dataset_generator.py


---
## 📬 Contact
For questions or suggestions, feel free to open an issue or contact the author:

Danielle Yaffe

yaffedan@post.bgu.ac.il

Research at Ben-Gurion University
