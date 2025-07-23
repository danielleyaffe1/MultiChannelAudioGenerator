import os
import json
import numpy as np
import soundfile as sf
import pyroomacoustics as pra
import random
import librosa
from tools import plot_room_2d, play_audio, plot_signal_at_microphones, get_gender_category
import sounddevice as sd
import matplotlib.pyplot as plt


class MultiChannelGenerator:
    def __init__(self, sample_rate=16000, output_dir=None, clean_output_dir=None, simulation='room', verbose=False, save_audio=True):
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.clean_output_dir = clean_output_dir
        if self.clean_output_dir:
            os.makedirs(clean_output_dir, exist_ok=True)
        self.verbose = verbose
        self.save_audio = save_audio
        self.microphone_array = np.array([
            [-0.082, -0.029, -0.005],  # Left temple
            [0.001, 0.030, -0.001],  # Above nose
            [0.077, 0.011, -0.002],  # Right temple
            [0.083, -0.060,-0.005],  # Inner right temple
        ])
        self.simulation_type = simulation

    def generate_room(self, room_dim, rt60_tgt, only_direct=False):
        """Creates a room with given size and RT60."""
        e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
        if self.simulation_type != 'room':
            max_order = 0

        if not only_direct:
            room = pra.ShoeBox(room_dim,
                               fs=self.sample_rate,
                               materials=pra.Material(e_absorption),
                               max_order=max_order,
                               use_rand_ism=False)

        else:   # To simulate only the direct signal (for DRR purposes)
            room = pra.ShoeBox(room_dim,
                               fs=self.sample_rate,
                               materials=pra.Material(e_absorption),
                               max_order=0,
                               use_rand_ism=False)

        return room

    def add_microphones(self, room, glasses_position):
        """Adds a microphone array to the room at glasses position."""
        mic_positions = self.microphone_array.T + np.array(glasses_position).reshape(3, 1)
        room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))
        return mic_positions

    def scale_noise(self, noise_signal, clean_signal, snr_db):
        """Scales noise to achieve target SNR."""
        clean_power = np.mean(clean_signal ** 2)
        noise_power = np.mean(noise_signal ** 2)
        snr_linear = 10 ** (snr_db / 10)
        return noise_signal * np.sqrt(clean_power / (snr_linear * noise_power))

    def add_speaker(self, room, glasses_position, audio, relative_position=[1.5, 90, 0], is_target=False, snr_db=None, interefering_audio=None):
        """Adds a speaker at a relative position to the listener."""
        r, phi, _ = relative_position
        position = glasses_position + np.array([r * np.cos(phi * np.pi / 180), r * np.sin(phi * np.pi / 180), 0])
        Lx, Ly, _ = room.shoebox_dim
        x, y = chk_speaker_position_in_room(position, Lx, Ly, glasses_position)
        position = np.array([x, y, position[2]])

        if is_target:
            room.add_source(position, signal=audio)
            return position, audio
        else:
            scaled_noise = self.scale_noise(interefering_audio, audio, snr_db)
            room.add_source(position, signal=scaled_noise)
            return position, scaled_noise

    def save_single_channel_audio(self, filename, data):
        """Saves a multi-channel audio file."""
        filename += '.wav'
        sf.write(filename, data.T, self.sample_rate)

    def save_multi_channel_audio(self, filename, signals):
        """Saves a multi-channel audio"""
        for i, signal in enumerate(signals):
            self.save_single_channel_audio(filename+f'_CH{i}', signal)
    def play_audio(self, signal):
        '''
        Play single audio
        :param signal: audio as a nparray
        :return:
        '''
        signal = signal / np.max(np.abs(signal))  # Normalize the audio signal to be within [-1, 1]

        # Convert the signal to 16-bit PCM format for better compatibility with sound devices
        signal = np.int16(signal * 32767)  # Scaling to the 16-bit integer range [-32767, 32767]

        # Play the audio signal
        sd.play(signal, self.sample_rate)
        sd.wait()  # Wait for playback to finish

    def generate_multichannel_audio(self, room_dim, rt60_tgt, snr, audio_files, num_interfering_sources=1, with_DRR=True):
        """
        Generates a multichannel audio with room size, RT60, and SNR condition with possible interfering sources.

        Parameters:
        - room_dim: Room dimension tuples (L, W, H).
        - rt60_tgt: RT60 value to simulate.
        - snr: SNR level for interfering speakers.
        - audio_files: dict, {"ID": int, "target": str with path, "interferes": dict with paths}.
        """

        sources_position = np.zeros((num_interfering_sources + 1, len(room_dim)))

        # Create room and microphone setup
        room = self.generate_room(room_dim, rt60_tgt)
        glasses_position = [room_dim[0] / 2, room_dim[1] / 2, 1.5] # michrophones set in the middle of the room
        mic_positions = self.add_microphones(room, glasses_position)

        # Add target speaker with random azimuth location
        target_audio, target_sr = librosa.load(audio_files["target"], sr=self.sample_rate)
        target_info = audio_files["target"].split('/')
        r_src, ph_src = 1.5, 90   # ph=0 is on the x+ axis, then pi increases counter clock wise
        target_position, _ = self.add_speaker(room, glasses_position, target_audio,
                                              [r_src, ph_src, 0], is_target=True)
        sources_position[0] = target_position

        if num_interfering_sources != len(audio_files["interferes"]):
            raise ValueError('Number of wanted interfering sources must match to generated target-interference definition')
        if self.verbose:
            print('Target:', audio_files["target"], 'Position:', target_position)
            print('Number of Interfering Sources:', num_interfering_sources)

        # Add interfering speakers with random azimuth location
        for n, spk in enumerate(audio_files["interferes"].keys()):
            interfering_audio, interfering_sr = librosa.load(audio_files["interferes"][spk], sr=self.sample_rate)
            r_noise, ph_noise = random.choice([3.0, 3.5]), random.randrange(180, 361, 20)   # ph=0 is on the x+ axis, then pi increases counter clock wise
            noise_pos, noise_sig = self.add_speaker(room, glasses_position, target_audio,
                                                    [r_noise, ph_noise, 0], is_target=False, snr_db=snr,
                                                    interefering_audio=interfering_audio)
            if self.verbose:
                print('Interfere:', audio_files["interferes"][spk], 'Position:', noise_pos)
            sources_position[n+1] = noise_pos

        # Simulate and get multi-channel signals
        room.simulate()
        signals = room.mic_array.signals

        filename = 'None'
        sample_id = audio_files["ID"]
        sentence = target_info[2].split('.')[0]
        # Save to file
        if self.output_dir:
            sampledir = self.output_dir #f"{self.output_dir}/{sample_id}/"
            os.makedirs(sampledir, exist_ok=True)
            filename = os.path.join(sampledir, f"target_{target_info[1]}_{sentence}_sampleid_{sample_id}")
            if self.save_audio:
                self.save_multi_channel_audio(filename, signals)

        drr_db = None

        if with_DRR or (self.clean_output_dir and self.save_audio):
            rir_full = room.rir[1][0]
            # create a room simulation with only the direct signal
            room_direct = self.generate_room(room_dim, rt60_tgt, only_direct=True)
            self.add_microphones(room_direct, glasses_position)
            self.add_speaker(room_direct, glasses_position, target_audio,[r_src, ph_src, 0], is_target=True)
            room_direct.simulate()
            rir_direct = room_direct.rir[1][0]  # RIR for mic 0 and source 0
            signals_direct = room_direct.mic_array.signals

            # Save to file
            if self.clean_output_dir and self.save_audio:
                sampledir = self.clean_output_dir #f"{self.clean_output_dir}/{sample_id}/"
                os.makedirs(sampledir, exist_ok=True)
                filename = os.path.join(sampledir,f"target_{target_info[1]}_{sentence}_sampleid_{sample_id}")
                self.save_multi_channel_audio(filename, signals_direct)

            if with_DRR:
                # Compute energy of direct path and reverberant path
                direct_energy = np.sum(rir_direct ** 2)
                reverb_energy = np.sum(rir_full ** 2) - direct_energy  # Remove direct energy from full RIR
                drr_db = 10 * np.log10(direct_energy / (reverb_energy + 1e-10))  # Avoid division by zero

            del rir_direct, room_direct, signals_direct

        rt60 = room.measure_rt60(plot=False) # measure the reverberation time, to show plot add plt.show() inside function

        if self.verbose:
            plot_room_2d(room, sources_position, mic_positions, T60=rt60_tgt, drr_dB=drr_db, sample_ID=sample_id, output_dir=None)
            print("The desired RT60 was {}, measured was {}".format(rt60_tgt, rt60[0][0]))
            plot_signal_at_microphones(signals, self.sample_rate)

        genders = get_gender_category(audio_files["target_id"], audio_files["interferer_ids"])

        return {"sample_id": sample_id,
                "target": audio_files["target"],
                "target_position": [round(x, 3) for x in target_position],
                "interferes": audio_files["interferes"],
                "interference_positions": [[round(x, 3) for x in pos] for pos in sources_position[1:]],
                "gender": genders,
                "room_dim": room_dim,
                "rt60": round(rt60[0][0], 1),
                "snr": snr,
                "DRR": float(round(drr_db, 3)),
                "num_channels": len(self.microphone_array),
                "file": filename
                }


def chk_speaker_position_in_room(position, Lx, Ly, glasses_position):
    '''check speaker position in room.
    Speaker should be 1 meter from all walls (checking only x,y plane). If not, position is fixed.
    :param position: Speaker position in room
    :param Lx: X room dimension
    :param Ly: Y room dimension
    '''
    if (Lx < 2.0) or (Ly < 2.0):
        raise ValueError(f"Room dimensions too small! Room is {Lx}x{Ly}, need at least 2m in both X and Y.")

    x, y, _ = position
    gx, gy, _ = glasses_position
    if x < 1.0 or x > Lx - 1.0 or y < 1.0 or y > Ly - 1.0:
        if x < 1.0:
            x = 1.0
        if x > Lx - 1.0:
            x = Lx - 1.0
        if y < 1.0:
            y = 1.0
        if y > Ly - 1.0:
            y = Ly - 1.0

        # Compute and print the radius (distance from center)
        radius = np.sqrt((x - gx) ** 2 + (y - gy) ** 2)
        if radius < 2.5:
            print(f"\nSpeaker at {position} must be ≥ 1 m from all walls, room is {Lx}x{Ly}")
            print(f"Fixing Position...Speaker new radial distance from Mics: {radius:.2f} meters")

    return x, y


