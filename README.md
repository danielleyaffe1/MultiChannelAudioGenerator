# 🎧 Multi-Channel Audio Dataset Generator

This repository generates a multi-channel audio datasets for speech and spatial audio research. It simulates reverberant rooms and microphone-array recordings using [Pyroomacoustics](https://github.com/LCAV/pyroomacoustics), and produces both mixed audio and clean/noise references for downstream tasks such as speech enhancement, source separation, and spatial audio modeling.

The pipeline is designed specifically for GRID Corpus audio-visual database structure. The database produces samples containing a target speaker plus one or more interfering speakers, with configurable room size, reverberation time, SNR, and microphone geometry flexibility.

---

## Main Workflow

1. Load target and interfering speech audio files.
2. Create room simulations with a configurable microphone array.
3. Generate reverberant target and interference signals.
4. Mix them at a chosen SNR and save:
   - multi-channel mixed audio,
   - multi-channel clean target audio,
   - multi-channel noise/interference-only audio,
   - optional: multi-channel direct-path target audio,
   - Not supported currently: multi-channel binaural audio.
5. Write metadata for each generated sample to a JSON file for training/evaluation pipelines.

---

## 🧩 Main components

- [dataset_generator.py](dataset_generator.py)
  - Entry point for generating datasets.
  - Defines the main generation parameters and runs the simulation loop.

- [roomaudiodatabase.py](roomaudiodatabase.py)
  - Implements the room acoustic simulation and multi-channel audio generation.
  - Includes microphone placement, source placement, RIR-based simulation, scaling, and saving.

- [tools/tools.py](tools/tools.py)
  - Helper functions for speaker-pair generation, plotting, and audio utilities.

- [tools/EDA.py](tools/EDA.py)
  - Generates exploratory data analysis plots from the generated metadata.

---

## 📁 Repository layout

```text
MultiChannelAudioGenerater/
├── dataset_generator.py          # Main script to run dataset generation
├── roomaudiodatabase.py         # Core room/acoustic simulation logic
├── tools/
│   ├── tools.py                 # Audio utilities and speaker-pair generation
│   └── EDA.py                   # Dataset statistics and visualization
├── cleanaudiodata/              # Input audio organized by speaker ID
├── cleandata/                   # Optional source audio/video data
├── speaker_pairs.json           # Generated speaker-target/interferer combinations
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🔧 Requirements

```bash
pip install -r requirements.txt
```

If you are using Conda, the repository also includes an environment file:

```bash
conda env create -f environment.yml
```

---

## 📥 Input data format

The generator expects input audio files to be stored in a speaker-organized folder structure like this:

```text
cleanaudiodata/
├── s1/
│   ├── sentence_1.wav
│   ├── sentence_2.wav
│   └── ...
├── s2/
│   └── ...
└── s34/
    └── ...
```

Each speaker folder should contain `.wav` files. The script builds target/interferer combinations from these folders automatically.

---

## ▶️ Quick start

1. Prepare your audio dataset under [cleanaudiodata](cleanaudiodata).
2. If you are starting from the audio-visual GRID Corpus dataset (Or a different audio-visual database with .mpg files), you need to extract the audio tracks from the `.mpg` files once. The repository includes a one-time (!) extraction step in [dataset_generator.py](dataset_generator.py). Use: 

```python
extract audio only files from .mpg files
extract_audio(AV_DATA_FOLDER, A_DATA_FOLDER)
print(f"Done extracting audio from mpg files in {AV_DATA_FOLDER}. Saved in {A_DATA_FOLDER}")
```

3. Open [dataset_generator.py](dataset_generator.py) and adjust the parameters at the top of the file. See below "Main Configuration Options" section
4. Run:

```bash
python dataset_generator.py
```

The script will generate samples and write metadata to JSON files in output folders such as:

- `Reverb_Babble/dataset.json`
- `Reverb_Babble/dataset`
- `Reverb_Babble/dataset_clean/`
---

## ⚙️ Main Configuration Options

The main parameters in [dataset_generator.py](dataset_generator.py) control the dataset generation:

- `snr_values`: list of target SNR values in dB
- `simulation`: `'room'` or `'free_field'`
- `noise`: `'interfere'` or `'babble'`
- `num_pairs_per_spk`: number of target sentences per speaker to use; set to `None` to use all available samples
- `max_total_samples`: optional limit on the total number of generated samples
- `save_multichannel`: save mixed multi-channel audio
- `save_noise_audio`: save interference-only audio
- `save_direct_audio`: save direct-path target audio
-  Not supported currently: `save_binaural`: optionally save binaural outputs
- `verbose`: print more detailed progress and diagnostics
- `EDA`: automatically generate dataset summary plots when enabled
- `room_sizes` and `rt60_values`: room dimensions and reverberation settings

The room simulation uses a glasses-mounted microphone array geometry from Aria V1 settings (Source: https://www.chimechallenge.org/challenges/chime8/task3/data) and places speakers at different relative positions inside the room.

---

## 📦 Output format

Audio outputs are written as `.wav` files, typically one folder per speaker, sentence and one file per channel (for example `S1/sentence/CH0`, `S1/sentence/CH1`, etc.).

Each generated sample is stored with metadata such as:

- target speaker ID,
- target sentence,
- interfering speaker IDs,
- room dimensions,
- RT60,
- SNR,
- sample identifier,
- output file paths.

---

## 📊 EDA and visualization

When `EDA = True`, the repository generates summary plots from the metadata JSON file, including:

- gender distribution,
- room size distribution,
- RT60 distribution,
- SNR distribution,
- samples per speaker,
- gender-combination statistics.

These plots are saved into the output plots directory for quick inspection.

---

## 📝 Notes

- The generator uses random room configurations and random interfering positions to increase diversity.
- If a `speaker_pairs.json` file already exists, the script reuses it and avoids regenerating the same samples unless the output metadata is missing or incomplete.
- The current implementation assumes a 16 kHz sample rate and a fixed simulation length of 3 seconds.
- Binaural ignals are not currently supported.

---

## 📬 Contact

For questions or suggestions, feel free to contact:

Danielle Yaffe  
yaffedan@post.bgu.ac.il

Research at Ben-Gurion University
