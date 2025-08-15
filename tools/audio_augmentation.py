import numpy as np
import random
import librosa
import soundfile as sf
import os


def load_random_audio_segment(folder, segment_length_s, sr):
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    assert files, "No WAV files in folder"
    while True:
        path = os.path.join(folder, random.choice(files))
        try:
            audio, file_sr = sf.read(path)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)  # convert to mono
            if file_sr != sr:
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
            break
        except Exception as e:
            print(f"Error reading {path}: {e}")
    segment_len = int(segment_length_s * sr)
    if len(audio) < segment_len:
        audio = np.pad(audio, (0, segment_len - len(audio)))
        start = 0
    else:
        start = random.randint(0, len(audio) - segment_len)
    return audio[start:start + segment_len]


def augment_audio(audio, sr, apply_all=False):
    """
    Apply a randomized combination of common augmentations.
    Set `apply_all=True` to always apply all.
    """

    def random_silence(audio, max_silence_s=0.5):
        silence_len = random.randint(0, int(sr * max_silence_s))
        silence = np.zeros(silence_len)
        insert_at = random.randint(0, len(audio))
        return np.concatenate([audio[:insert_at], silence, audio[insert_at:]])

    def volume(audio, min_gain=0.8, max_gain=1.2):
        return audio * random.uniform(min_gain, max_gain)

    def time_stretch(audio, min_rate=0.9, max_rate=1.1):
        rate = random.uniform(min_rate, max_rate)
        try:
            return librosa.effects.time_stretch(audio, rate)
        except:
            return audio  # fails on short signals

    def pitch_shift(audio, max_semitones=2):
        steps = random.uniform(-max_semitones, max_semitones)
        try:
            return librosa.effects.pitch_shift(audio, sr, n_steps=steps)
        except:
            return audio

    def chunk_shuffle(audio, num_chunks=3):
        if len(audio) < num_chunks:
            return audio
        chunk_size = len(audio) // num_chunks
        chunks = [audio[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
        random.shuffle(chunks)
        return np.concatenate(chunks)

    # Apply augmentations randomly or always
    if apply_all or random.random() < 0.5:
        audio = random_silence(audio)
    if apply_all or random.random() < 0.5:
        audio = volume(audio)
    if apply_all or random.random() < 0.3:
        audio = time_stretch(audio)
    if apply_all or random.random() < 0.3:
        audio = pitch_shift(audio)
    if apply_all or random.random() < 0.5:
        audio = chunk_shuffle(audio)

    # Pad/crop to fixed length if needed
    return audio[:len(audio)]  # optional normalization


# Example usage
if __name__ == "__main__":
    sr = 16000
    segment_len = 2.5  # seconds
    folder = "/path/to/speech_clips"

    audio = load_random_audio_segment(folder, segment_length_s=segment_len, sr=sr)
    augmented = augment_audio(audio, sr)

    sf.write("augmented_example.wav", augmented, sr)
