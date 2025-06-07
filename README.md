# 🎧 Multi-Channel Audio Dataset Generator

This repository provides a tool to generate multi-channel audio datasets using simulated room acoustics and microphone arrays. It is designed for research in audio enhancement, speech separation, and spatial audio modeling. The tool simulates reverberant environments with configurable source and microphone positions using [Pyroomacoustics](https://github.com/LCAV/pyroomacoustics).

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


---
## 🔧 Requirements
pip install -r requirements.txt


---
## 🧪 Example Usage
Edit dataset_generator.py:
  - Room size: list of [x,y,z] m^3 configurations
  - RT60: list of t60 values
  - SNR: list of SNR values (dB)
  - Number of interfering speakers
  - Number of samples for each target speaker
  - Input audio paths

Run: python dataset_generator.py


---
## 📬 Contact
For questions or suggestions, feel free to open an issue or contact the author:

Danielle Yaffe

yaffedan@post.bgu.ac.il

Research at Ben-Gurion University
