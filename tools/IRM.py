import numpy as np
import soundfile as sf
import librosa
import json
import os
import sys
p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)
from roomaudiodatabase import MultiChannelGenerator

def mix_with_snr(speech, noise, snr_db):
    """
    Mix speech and noise at a target SNR in the time domain.
    """
    # Match lengths
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
    noise = noise[:len(speech)]

    # Compute current powers
    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)

    # Compute scaling factor for noise
    target_noise_power = speech_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)

    # Mix
    mixture = speech + noise
    return mixture, noise

def compute_irm(speech, noise, n_fft=512, hop_length=128):
    """
    Compute Ideal Ratio Mask (IRM) from speech and noise.
    """
    S = librosa.stft(speech, n_fft=n_fft, hop_length=hop_length)
    N = librosa.stft(noise, n_fft=n_fft, hop_length=hop_length)

    mag_S = np.abs(S)
    mag_N = np.abs(N)

    irm = mag_S**2 / (mag_S**2 + mag_N**2 + 1e-8)  # Avoid division by zero
    return irm

def compute_cirm(speech, mixture, n_fft=512, hop_length=128):
    """
    Compute Complex Ideal Ratio Mask (cIRM) from speech and mixture.
    """
    S = librosa.stft(speech, n_fft=n_fft, hop_length=hop_length)
    X = librosa.stft(mixture, n_fft=n_fft, hop_length=hop_length)

    cirm = S / (X + 1e-8)  # Avoid division by zero
    return cirm

def apply_mask(mixture, mask, n_fft=512, hop_length=128):
    """
    Apply the given mask to the mixture in STFT domain.
    """
    M = librosa.stft(mixture, n_fft=n_fft, hop_length=hop_length)
    enhanced_mag = np.abs(M) * mask
    enhanced = librosa.istft(enhanced_mag * np.exp(1j * np.angle(M)), hop_length=hop_length)
    return enhanced


if __name__ == "__main__":
    simualtion_name = 'Reverb_Babble'
    simualtion_metadata_path = os.path.join(p,simualtion_name)

    simulation, noise = simualtion_name.split('_')
        
    audio_generator = MultiChannelGenerator(sample_rate=16000,
                                              output_dir=output_dir,
                                              clean_output_dir=clean_output_dir,
                                              simulation=simulation,
                                              verbose=False)
    
    
    with open(simualtion_metadata_path) as f:
        for line in f:
            meta_data= json.loads(line)
    
    # Load your clean speech and noise (mono, same sample rate)
    speech, sr = librosa.load("clean_speech.wav", sr=None)
    #noise, _ = librosa.load("noise.wav", sr=sr)
    noise = np.random.normal(0.0, 1.0, len(speech))

    # Mix at 5 dB SNR
    mixture, noise_scaled = mix_with_snr(speech, noise, snr_db=-10)

    # Compute IRM
    irm = compute_irm(speech, noise_scaled)

    # Apply IRM to noisy mixture
    enhanced = apply_mask(mixture, irm)

    # Save outputs
    sf.write("mixture.wav", mixture, sr)
    sf.write("enhanced_irm.wav", enhanced, sr)

    print("Done! Now you can run PESQ/STOI/SI-SDR on enhanced_irm.wav.")
