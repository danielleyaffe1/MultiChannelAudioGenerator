
from moviepy.editor import VideoFileClip
import soundfile as sf
import pygame as pg
pg.init()
pg.display.set_caption('MoviePy')
import os
from tools import *
from room_generator import *



if __name__ == "__main__":
    # Load the video file
    video_path = "cleandata/s1/bbaf2n.mpg"
    video = VideoFileClip("cleandata/s1/bbaf2n.mpg")

    # Extract audio
    audio = video.audio
    sample_rate = audio.fps
    #audio.preview(fps=sample_rate)
    print(f"Audio duration: {audio.duration}, Sampling Rate: {sample_rate}")

    # TODO: for now save .wav file and then work with it. having problems with getting the np.array audio with AudioFileClip
    #sndarray, fps = extract_audio(clip=audio, fps=sample_rate)
    audio_path = video_path.replace("cleandata", "cleanaudiodata")
    audio_path = audio_path.replace(".mpg", ".wav")
    if not os.path.exists(audio_path):
        audio.write_audiofile(audio_path, fps=sample_rate, codec="pcm_s16le")

    audio, sample_rate = librosa.load(audio_path, mono=True)

    # TODO: would want to vary acros a few values of room_dim, rt60_tgt, source_position: azimuth=[60,120]
    # source location

    r_src, ph_src, th_src = 1.0, 90, 0  #source radios (m), azimuth(degree), altitude(degree)
    relative_source_position = [r_src, ph_src, th_src]
    mic_signals = room_generator(audio,
                   sample_rate,
                   room_dim=[6.0, 5.0, 3.0], #[length, width, height],
                   rt60_tgt=0.4,  # desired reverberation time [seconds]
                   relative_source_position=relative_source_position,
                   with_prev=False)

    #signal_at_microphones(mic_signals, sample_rate)
    #signal_at_microphones(np.expand_dims(audio,axis=0), sample_rate)
    # Assuming `mic_signals` is the output of room_generator (shape: num_mics x num_samples)
    mic_energies = np.sum(mic_signals ** 2, axis=1)

    # Print the energy of each mic
    for i, energy in enumerate(mic_energies):
        print(f"Microphone {i + 1} Energy: {energy:.2f}")

    # Play original signal
    # print("Playing original signal:")
    # play_audio(audio, sample_rate)

    # Play simulated signals from each microphone
    # for i, mic_signal in enumerate(mic_signals):
    #     print(f"Playing signal from Microphone {i + 1}:")
    #     play_audio(mic_signal, sample_rate)
        # output_path = f"/Users/danielleyaffe/Desktop/audio_simulations/Glasses array - pyroomacoustics/bbaf2n_mic_{i + 1}.wav"
        # sf.write(output_path, mic_signal, sample_rate)
        # print(f"Microphone {i + 1} signal saved to {output_path}")



