import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import librosa
import sounddevice as sd
import pygame as pg
from moviepy.editor import VideoFileClip
import os
import random
pg.init()
pg.display.set_caption('MoviePy')


def plot_time_domain(signal, sr):

    # Create a plot for the time-domain signal
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(signal)) / sr, signal, label="Signal", color='b')
    plt.title("Time Domain Signal")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_stft(signal, sr):

    # Compute the STFT of the signal
    D = librosa.stft(signal)
    # Convert the amplitude to decibels
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Create a plot for the STFT (spectrogram)
    plt.figure(figsize=(10, 6))
    img = librosa.display.specshow(D_db, x_axis='time', y_axis='log', sr=sr)
    plt.title("STFT Magnitude (Log Frequency Scale)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(format="%+2.0f dB")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def check_reverberation(signal, fs):
    # Cross-correlate the signal with itself
    correlation = correlate(signal, signal, mode='full')
    lags = np.arange(-len(signal) + 1, len(signal))

    # Normalize the correlation for better visualization
    correlation /= np.max(np.abs(correlation))

    # Plot the correlation
    plt.figure(figsize=(10, 4))
    plt.plot(lags / fs, correlation)
    plt.title("Matched Filter Output (Self-Correlation)")
    plt.xlabel("Time Lag (seconds)")
    plt.ylabel("Normalized Correlation")
    plt.grid()
    plt.show()


def normalize_signal(signal):
    return signal / np.max(np.abs(signal))


def play_audio(signal, samplerate):
    '''
    Play single audio
    :param signal: audio as a nparray
    :return:
    '''
    signal = normalize_signal(signal)  #Normalize the audio signal to be within [-1, 1]

    # Convert the signal to 16-bit PCM format for better compatibility with sound devices
    signal = np.int16(signal * 32767)  # Scaling to the 16-bit integer range [-32767, 32767]

    # Play the audio signal
    sd.play(signal, samplerate)
    sd.wait()  # Wait for playback to finish


def extract_audio(av_folder, audioonly_folder):
    speaker_dirs = os.listdir(av_folder)
    for speaker_dir in speaker_dirs:
        av_file_count = 0
        if 'DS_Store' in speaker_dir:
            os.remove(os.path.join(av_folder, speaker_dir))
            continue
        speaker_files = os.listdir(av_folder + '/' + speaker_dir)
        if not os.path.exists(os.path.join(audioonly_folder, speaker_dir)):
            os.makedirs(os.path.join(audioonly_folder, speaker_dir))

        for file in speaker_files:
            if not file.endswith('.mpg'):
                print(speaker_dir, file)
                continue
            # Load the video file
            av_file_count +=1

            video_path = os.path.join(av_folder, speaker_dir, file)
            video = VideoFileClip(video_path)

            # Extract audio
            audio = video.audio
            sample_rate = audio.fps
            #audio.preview(fps=sample_rate)
            #print(f"Audio duration: {audio.duration}, Sampling Rate: {sample_rate}")

            audio_path = video_path.replace(av_folder, audioonly_folder)
            audio_path = audio_path.replace(".mpg", ".wav")
            if not os.path.exists(audio_path):
                audio.write_audiofile(audio_path, fps=sample_rate, codec="pcm_s16le", logger=None)

        if av_file_count == len(os.listdir(os.path.join(av_folder, speaker_dir))):
            print(f'All .mpg files from speaker {speaker_dir} have been saved as .wav files')


def plot_signal_at_microphones(mic_signals, fs, start=0):
    # plot signal at microphones
    plt.figure(figsize=(15, 10))
    step_size = 0.2  # X-axis tick interval
    for i in range(mic_signals.shape[0]):
        plt.subplot(4, 1, i + 1)
        end = len(mic_signals[i])
        time_axis = np.arange(start, end) / fs  # Time in seconds
        mic_signal = mic_signals[i, start:end]
        #plt.plot(np.arange(len(mic_signal)) / fs, mic_signal)
        plt.plot(time_axis, mic_signal)
        plt.title("Microphone {} Signal".format(i + 1))
        plt.xlabel("Time [s]")
        plt.xticks(np.arange(time_axis[0], time_axis[-1], step_size))
        #plt.yticks(np.arange(-1, 1.1, 0.5))  # Step by 0.1

        plt.grid(True)

    plt.subplots_adjust(hspace=1.3)  # Increase vertical space
    plt.show()


def plot_room_2d(room, source_position, mic_positions, sample_ID, T60=None, drr_dB=None, output_dir=None, directivity=False, azimuth_deg=None):
    """
    Plots a 2D representation of the room with the speaker and microphone positions.

    Parameters:
    - room: pyroomacoustics.ShoeBox object (room definition)
    - source_position: np.array of shape (2,) or (2, N) representing source positions (x, y)
    - mic_positions: np.array of shape (2, N) representing microphone positions (x, y)
    - sample_ID: identifier for the sample
    - T60: RT60 value (optional)
    - drr_dB: Direct-to-reverberant ratio in dB (optional)
    - output_dir: directory to save the plot
    - directivity: bool, whether to draw directivity arrows
    - azimuth: float, azimuth angle in degrees for shared directivity
    """
    if directivity and azimuth_deg is None:
        raise ValueError("If directivity is chosen to be presented in the plot, an azimuth angle must be provided")
    source_positionT = source_position.T
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, room.shoebox_dim[0])
    ax.set_ylim(0, room.shoebox_dim[1])

    # Plot speaker position
    labels = ['Speaker'] + ['Interferer'] * (len(source_position) - 1)
    colors = ['green'] + ['red'] * (len(source_position) - 1)
    for i, source in enumerate(source_position):
        ax.scatter(source_positionT[0][i], source_positionT[1][i], color=colors[i], marker='o', s=50, label=labels[i])

    # Plot microphone positions
    ax.scatter(mic_positions[0], mic_positions[1], marker='*', s=10, label='Microphones')

    # Plot a single directivity arrow
    if directivity:
        # Use center point between microphones for arrow origin
        center_x = np.mean(mic_positions[0])
        center_y = np.mean(mic_positions[1])
        azimuth_rad = np.deg2rad(azimuth_deg)
        dx = 0.3 * np.cos(azimuth_rad)
        dy = 0.3 * np.sin(azimuth_rad)
        ax.arrow(center_x, center_y, dx, dy, head_width=0.2, head_length=0.2,
                 fc='blue', ec='blue', label=f'Directivity ({azimuth_deg}°)')

    # Labels and legend
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    title = f"Room Simulation: {sample_ID}\n Room Size: {room.shoebox_dim[0]}m x {room.shoebox_dim[1]}m"
    if T60 is not None:
        title += f" | RT60: {T60}s"
    if drr_dB is not None:
        title += f" | DRR: {drr_dB:.2f} dB"
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    if output_dir:
        filename = os.path.join(output_dir, 'room_simulation.png')
        plt.savefig(filename)
    else:
        plt.show()

    # Determine zoomed-in region
    zoom_margin = 0.1 #meters
    min_x = min(np.min(source_positionT[0]), np.min(mic_positions[0])) - zoom_margin
    max_x = max(np.max(source_positionT[0]), np.max(mic_positions[0])) + zoom_margin
    min_y = min(np.min(source_positionT[1]), np.min(mic_positions[1])) - zoom_margin
    max_y = max(np.max(source_positionT[1]), np.max(mic_positions[1])) + zoom_margin

    # Zoomed-in plot
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.set_xlim(min_x, max_x)
    ax2.set_ylim(min_y, max_y)

    # Plot speaker and microphones
    for i, source in enumerate(source_position):
        ax2.scatter(source_positionT[0][i], source_positionT[1][i], color=colors[i], marker='o', s=50, label=labels[i])

    if directivity:
        labels = ['Left', 'Right']
        ax2.arrow(center_x, center_y, dx, dy, head_width=0.2, head_length=0.05,
                  fc='blue', ec='blue', label=f'Directivity ({azimuth_deg}°)')
        for i, label in enumerate(labels):
            ax2.scatter(mic_positions[0][0], mic_positions[1][0], marker='*', s=15, label=labels[i])

    else:
        labels = ['Left temple', 'Above nose', 'Right temple', 'Inner right temple']
        for i, label in enumerate(labels):
            ax2.scatter(mic_positions[0][i], mic_positions[1][i], marker='*', s=15, label=labels[i])

    ax2.set_xlabel("X (meters)")
    ax2.set_ylabel("Y (meters)")
    title = f"Zoomed-in: Speaker & Microphones for Sample ID {sample_ID}\n Room Size: {room.shoebox_dim[0]}m x {room.shoebox_dim[1]}m"
    if T60 is not None:
        title += f" | RT60: {T60}s"
    if drr_dB is not None:
        title += f" | DRR: {drr_dB:.2f} dB"
    ax2.set_title(title)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if output_dir:
        filename = os.path.join(output_dir, 'room_simulation_zoom.png')
        plt.savefig(filename)
    else:
        plt.show()


def generate_speaker_pairs(datapath, num_samples=None, num_interferers=1, max_total_samples=None):
    """
    Generates `num_samples` target-interferer pairs per speaker.
    - Each speaker is a target `num_samples` times.
    - Sentences are stored as file paths.
    - Interferes are chosen dynamically (excluding the target).
    """
    audio_files = {}
    for speaker in os.listdir(datapath):
        if speaker.endswith('.DS_Store'):
            continue
        audio_files[speaker] = os.listdir(os.path.join(datapath, speaker))

    male_id = ['s1', 's2', 's3', 's5', 's6', 's8', 's9', 's10', 's12', 's13', 's14', 's17', 's19', 's26', 's27', 's28',
               's30', 's32']
    female_id = ['s4', 's7', 's11', 's15', 's16', 's18', 's20', 's22', 's23', 's24', 's25', 's29', 's31', 's33', 's34']
    speaker_pairs = []
    sample_id = 0
    for target_speaker in audio_files.keys():
        if num_samples is None:
            num_samples = len(audio_files[target_speaker])
        for target_sentence in audio_files[target_speaker][0:num_samples]:

            # Interfering speakers (excluding target)
            available_speakers = [spk for spk in audio_files.keys() if spk != target_speaker]
            # Exclude same gender as well
            # if target_speaker in male_id:
            #     available_speakers = [spk for spk in audio_files.keys() if spk != target_speaker and spk in female_id]
            # elif target_speaker in female_id:
            #     available_speakers = [spk for spk in audio_files.keys() if spk != target_speaker and spk in male_id]
            # else:
            #     raise ValueError(f"Unknown gender for target speaker {target_speaker}")

            interferes = random.sample(available_speakers, min(num_interferers, len(available_speakers)))

            # Select sentences for interferes
            interferer_sentences = {spk: os.path.join(datapath, spk, random.choice(audio_files[spk])) for spk in interferes}
            
            speaker_pairs.append({
                "ID": sample_id,
                "target": os.path.join(datapath, target_speaker, target_sentence),
                "target_id": target_speaker,
                "interferes": interferer_sentences,
                "interferer_ids": interferes
            })
            sample_id += 1
            if (max_total_samples is not None) and (sample_id == max_total_samples):
                return speaker_pairs

    return speaker_pairs


def get_gender_category(target_id, interferer_ids):
    """
    Returns the gender category of the speakers in the simulation.

    Args:
        target_id (str): ID of the target speaker, e.g., 's1'.
        interferer_ids (list of str): List of interfering speaker IDs.

    Returns:
        str: One of 'male', 'female', or 'mixed'.
    """

    male_id = ['s1', 's2', 's3', 's5', 's6', 's8', 's9', 's10', 's12', 's13', 's14', 's17', 's19', 's26', 's27', 's28',
               's30', 's32']
    female_id = ['s4', 's7', 's11', 's15', 's16', 's18', 's20', 's22', 's23', 's24', 's25', 's29', 's31', 's33', 's34']

    def get_gender(speaker_id):
        if speaker_id in male_id:
            return 'male'
        elif speaker_id in female_id:
            return 'female'
        else:
            raise ValueError(f"Unknown speaker ID: {speaker_id}")


    all_ids = [target_id] + interferer_ids
    genders = {get_gender(sid) for sid in all_ids}
    if len(genders) == 1:
        return genders.pop()  # 'male' or 'female'
    elif len(genders) == 2:
        return 'mixed'
    else:
        raise ValueError(f"Unknown genders for speaker ID: {all_ids}")
