import os
import json
import numpy as np
import soundfile as sf
import pyroomacoustics as pra
import random
import librosa
from tools.tools import plot_room_2d, play_audio, plot_signal_at_microphones, get_gender_category, plot_stft
import matplotlib.pyplot as plt
from pyroomacoustics.directivities import MeasuredDirectivityFile, Rotation3D


class MultiChannelGenerator:
    def __init__(self, sample_rate=16000, output_dir=None, clean_output_dir=None, noise_output_dir=None, simulation='room', verbose=False, verbose_outpur_dir=None):
        self.sample_rate = sample_rate
        self.audio_length = 3   # sec
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.clean_output_dir = clean_output_dir
        if self.clean_output_dir:
            os.makedirs(clean_output_dir, exist_ok=True)
        self.noise_output_dir = noise_output_dir
        if self.noise_output_dir:
            os.makedirs(noise_output_dir, exist_ok=True)
        self.verbose = verbose
        self.verbose_outpur_dir = verbose_outpur_dir
        if self.verbose_outpur_dir:
            os.makedirs(verbose_outpur_dir, exist_ok=True)
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
                               max_order=40, # FIXME
                               use_rand_ism=False)

        else:   # To simulate only the direct signal
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

    def scale_noise(self, noise_signal, clean_signal, snr_db, find_alpha=False):
        """Scales noise to achieve target SNR."""
        clean_power = np.mean(clean_signal ** 2)
        noise_power = np.mean(noise_signal ** 2)
        snr_linear = 10 ** (snr_db / 10)
        
        if find_alpha:
            return np.sqrt(clean_power / (snr_linear * noise_power))
        
        return noise_signal * np.sqrt(clean_power / (snr_linear * noise_power))

    def add_speaker(self, room, glasses_position, audio, relative_position=[1.5, 90, 0], need_scaling=False, snr_db=None, interefering_audio=None):
        """Adds a speaker at a relative position to the listener."""
        r, phi, _ = relative_position
        position = glasses_position + np.array([r * np.cos(phi * np.pi / 180), r * np.sin(phi * np.pi / 180), 0])
        Lx, Ly, _ = room.shoebox_dim
        x, y = chk_speaker_position_in_room(position, Lx, Ly, glasses_position)
        position = np.array([x, y, position[2]])

        if not need_scaling:
            room.add_source(position, signal=audio)
            return position, audio
        else:
            scaled_noise = self.scale_noise(interefering_audio, audio, snr_db)
            room.add_source(position, signal=scaled_noise)
            return position, scaled_noise

    def crop_audio_length(self, audio):
        segment_len = int(self.audio_length * self.sample_rate)
        if len(audio) < segment_len:
            # Pad with zeros if file is too short
            return np.pad(audio, (0, segment_len - len(audio)))

        return audio[:segment_len]

    def save_single_channel_audio(self, filename, data):
        """Saves a multi-channel audio file."""
        filename += '.wav'
        sf.write(filename, data.T, self.sample_rate)

    def save_multi_channel_audio(self, filename, signals):
        """Saves a multi-channel audio"""
        for i, signal in enumerate(signals):
            cropped_signal = self.crop_audio_length(signal)
            self.save_single_channel_audio(filename+f'_CH{i}', cropped_signal)

    def save_binaural_audio(self, filename, signals):
        """Saves a binaural audio
            Expecting signals[0] to be the left ear and signals[1] to be the right"""
        if signals.shape[0] != 2:
            raise ValueError('Input to save_binaural_audio must be left and right ear signals')
        
        cropped_signals = np.zeros((2, int(self.audio_length * self.sample_rate)))
        for i, sig in enumerate(signals):
            cropped_signals[i] = self.crop_audio_length(sig)
        sf.write(filename+'.wav', cropped_signals.T, self.sample_rate)


    def generate_multichannel_audio(self, room_dim, rt60_tgt, snr, audio_files, num_interfering_sources=1, with_DRR=False, save_audio=False, save_noise_audio=False):
        """
        Generates a multichannel audio with room size, RT60, and SNR condition with possible interfering sources.

        Parameters:
        - room_dim: Room dimension tuples (L, W, H).
        - rt60_tgt: RT60 value to simulate.
        - snr: SNR level for interfering speakers [dB].
        - audio_files: dict, {"ID": int, "target": str with path, "interferes": dict with paths}.
        """

        source_positions = np.zeros((num_interfering_sources + 1, len(room_dim)))
        glasses_position = [room_dim[0] / 2, room_dim[1] / 2, 1.5] # michrophones set in the middle of the room
        
        # Room simulation with only the target speaker (For SNR scaling and saving target data)
        room_only_target = self.generate_room(room_dim, rt60_tgt)
        mic_positions = self.add_microphones(room_only_target, glasses_position)
        
        # Add target speaker with random azimuth location
        target_audio, target_sr = librosa.load(audio_files["target"], sr=self.sample_rate)
        target_info = audio_files["target"].split('/')
        r_src, ph_src = 0.5, 90   # ph=0 is on the x+ axis, then pi increases counter clock wise
        target_position, _ = self.add_speaker(room_only_target, glasses_position, target_audio,[r_src, ph_src, 0], need_scaling=False)
        
        source_positions[0] = target_position

        room_only_target.simulate()
        rir_only_target = room_only_target.rir[1][0]  # RIR for mic 0 and source 0 (later used for DRR)
        signals_only_target = room_only_target.mic_array.signals
        
        
        # Room simulation with only interfers/ noise speakers (for SNR scaling and saving noise data)
        if num_interfering_sources != len(audio_files["interferes"]):
            raise ValueError('Number of wanted interfering sources must match to generated target-interference definition')
        
        room_only_interfering = self.generate_room(room_dim, rt60_tgt)
        self.add_microphones(room_only_interfering, glasses_position)

        # Add interfering speakers with random azimuth location
        for n, spk in enumerate(audio_files["interferes"].keys()):
            interfering_audio, _ = librosa.load(audio_files["interferes"][spk], sr=self.sample_rate)
            # interfering_audio = self.crop_audio_length(interfering_audio) #think of adding audio augmentation
            r_noise, ph_noise = random.choice([2.5, 3.5]), random.randrange(180, 361, 20)   # ph=0 is on the x+ axis, then pi increases counter clock wise
            noise_pos, _ = self.add_speaker(room_only_interfering, glasses_position, interfering_audio,[r_noise, ph_noise, 0], need_scaling=False)
            source_positions[n+1] = noise_pos
            
            if self.verbose:
                print('Interfere:', audio_files["interferes"][spk], 'Position:', noise_pos)
            
        room_only_interfering.simulate()
        signals_only_interfering = room_only_interfering.mic_array.signals

        # Simulate multi-channel signals with SNR
        alpha = self.scale_noise(signals_only_interfering[0], signals_only_target[0], snr, find_alpha=True) #without loss of generalization compute SNR according to channel 0
        signals = signals_only_target[:,:int(self.audio_length * self.sample_rate)] + alpha*signals_only_interfering[:,:int(self.audio_length * self.sample_rate)]

        # Saving data: mixture data
        filename = 'None'
        sample_id = audio_files["ID"]
        sentence = target_info[2].split('.')[0]
        # Save to file
        if self.output_dir:
            filename = os.path.join(self.output_dir, f"target_{target_info[1]}_{sentence}_sampleid_{sample_id}")
            if save_audio:
                self.save_multi_channel_audio(filename, signals)

        # Saving data: clean data
        if save_audio:
            if self.clean_output_dir is None:
                raise ValueError('A clean output directory has not been provided')
            filename = os.path.join(self.clean_output_dir, f"target_{target_info[1]}_{sentence}_sampleid_{sample_id}")
            self.save_multi_channel_audio(filename, signals_only_target)

        # Saving data: noise data
        if save_noise_audio:
            if self.noise_output_dir is None:
                raise ValueError('A noise output directory has not been provided')
            filename = os.path.join(self.noise_output_dir, f"target_{target_info[1]}_{sentence}_sampleid_{sample_id}")
            self.save_multi_channel_audio(filename, alpha*signals_only_interfering)

        drr_db = None
        if with_DRR:
            window_ms = 1.0
            peak_idx = np.argmax(np.abs(rir_only_target))
            window_samples = int(window_ms * self.sample_rate / 1000)

            start = max(0, peak_idx - window_samples)
            end = min(len(rir_only_target), peak_idx + window_samples)

            rir_direct = rir_only_target[start:end]
            rir_reverb = np.concatenate((rir_only_target[:start], rir_only_target[end:]))

            direct_energy = np.sum(rir_direct ** 2)
            reverb_energy = np.sum(rir_reverb ** 2)

            drr_db = 10 * np.log10(direct_energy / (reverb_energy + 1e-10))  # Avoid division by zero
            #room_only_target.plot_rir(FD=False)

        rt60 = room_only_target.measure_rt60(plot=False) # measure the reverberation time, to show plot add plt.show() inside function

        if self.verbose:
            print('Target:', audio_files["target"], 'Position:', target_position)
            print('Number of Interfering Sources:', num_interfering_sources)
            plot_room_2d(room_only_target, source_positions, mic_positions, sample_ID=sample_id, T60=round(rt60[0][0], 1),
                         drr_dB=drr_db, SNR=snr, output_dir=self.verbose_outpur_dir)
            print("The desired RT60 was {}, measured was {}".format(rt60_tgt, rt60[0][0]))
            print(f'DRR DB: {drr_db}')
            plot_signal_at_microphones(signals, self.sample_rate, output_dir=self.verbose_outpur_dir)
            plot_stft(signal=signals[0], sr=self.sample_rate, output_dir=self.verbose_outpur_dir, signal_name='mixture')
            plot_stft(signal=signals_only_target[0], sr=self.sample_rate, output_dir=self.verbose_outpur_dir, signal_name='target')
            plot_stft(signal=signals_only_interfering[0], sr=self.sample_rate, output_dir=self.verbose_outpur_dir, signal_name='noise')

        genders = get_gender_category(audio_files["target_id"], audio_files["interferer_ids"])

        return {"sample_id": sample_id,
                "target": audio_files["target"],
                "target_position": [round(x, 3) for x in target_position],
                "interferes": audio_files["interferes"],
                "interference_positions": [[round(x, 3) for x in pos] for pos in source_positions[1:]],
                "room_dim": room_dim,
                "rt60": round(rt60[0][0], 1),
                "snr": snr,
                "DRR": float(round(drr_db, 3)),
                "num_channels": len(self.microphone_array),
                "gender": genders,
                "file": filename
                }

    def generate_binaural_audio(self, room_dim, rt60_tgt, snr, audio_files, num_interfering_sources=1,
                                azimuth_deg=-45.0, colatitude_deg=90.0, save_audio=True):

        source_positions = np.zeros((num_interfering_sources + 1, len(room_dim)))

        # Create room and microphone setup
        room = self.generate_room(room_dim, rt60_tgt)
        glasses_position = [room_dim[0] / 2, room_dim[1] / 2, 1.5]  # michrophones set in the middle of the room

        # Add target speaker with random azimuth location
        target_audio, target_sr = librosa.load(audio_files["target"], sr=self.sample_rate)
        # target_audio = self.crop_audio_length(target_audio)
        #target_audio = target_audio * (0.95 / abs(target_audio).max())

        hrtf = MeasuredDirectivityFile(
            path='mit_kemar_normal_pinna.sofa',
            fs=self.sample_rate,
            interp_order=12,
            interp_n_points=1000,
        )
        orientation = Rotation3D([colatitude_deg, azimuth_deg], "yz", degrees=True)
        dir_left = hrtf.get_mic_directivity("left", orientation=orientation)
        dir_right = hrtf.get_mic_directivity("right", orientation=orientation)

        room.add_microphone(np.array(glasses_position).reshape(3, 1), directivity=dir_left)
        room.add_microphone(np.array(glasses_position).reshape(3, 1), directivity=dir_right)

        target_info = audio_files["target"].split('/')
        r_src, ph_src = 0.5, 90  # ph=0 is on the x+ axis, then pi increases counter clock wise
        target_position, _ = self.add_speaker(room, glasses_position, target_audio,
                                              [r_src, ph_src, 0], is_target=True)
        source_positions[0] = target_position

        if num_interfering_sources != len(audio_files["interferes"]):
            raise ValueError(
                'Number of wanted interfering sources must match to generated target-interference definition')
        if self.verbose:
            print('Target:', audio_files["target"], 'Position:', target_position)
            print('Number of Interfering Sources:', num_interfering_sources)

        # Add interfering speakers with random azimuth location
        for n, spk in enumerate(audio_files["interferes"].keys()):
            interfering_audio, interfering_sr = librosa.load(audio_files["interferes"][spk], sr=self.sample_rate)
            # interfering_audio = self.crop_audio_length(interfering_audio) #think of adding audio augmentation
            r_noise, ph_noise = random.choice([2.5, 3.5]), random.randrange(180, 361,
                                                                            20)  # ph=0 is on the x+ axis, then pi increases counter clock wise
            noise_pos, noise_sig = self.add_speaker(room, glasses_position, target_audio,
                                                    [r_noise, ph_noise, 0], is_target=False, snr_db=snr,
                                                    interefering_audio=interfering_audio)
            if self.verbose:
                print('Interfere:', audio_files["interferes"][spk], 'Position:', noise_pos)
            source_positions[n + 1] = noise_pos

        # Simulate and get binaural signals
        room.simulate()
        binaural_signals = room.mic_array.signals
        #binaural_signals *= 0.95 / abs(binaural_signals).max()
        #binaural_signals = (binaural_signals * 2 ** 15).astype(np.int16)

        filename = 'None'
        sample_id = audio_files["ID"]
        sentence = target_info[2].split('.')[0]
        # Save to file
        if self.output_dir:
            sampledir = self.output_dir+'_Binaural'  # f"{self.output_dir}/{sample_id}/"
            os.makedirs(sampledir, exist_ok=True)
            filename = os.path.join(sampledir, f"target_{target_info[1]}_{sentence}_sampleid_{sample_id}")
            if save_audio:
                self.save_binaural_audio(filename, binaural_signals)

        rt60 = room.measure_rt60(
            plot=False)  # measure the reverberation time, to show plot add plt.show() inside function

        if self.verbose:
            plot_room_2d(room, source_positions, np.array(glasses_position).reshape(3, 1), sample_ID=sample_id,
                         T60=round(rt60[0][0], 1), output_dir=self.verbose_outpur_dir,directivity=True, azimuth_deg=azimuth_deg)
            print("The desired RT60 was {}, measured was {}".format(rt60_tgt, rt60[0][0]))
            room.plot_rir(FD=True)
            room.plot_rir(FD=False)
            plot_signal_at_microphones(binaural_signals, self.sample_rate, output_dir=self.verbose_outpur_dir, binaural=True)

        genders = get_gender_category(audio_files["target_id"], audio_files["interferer_ids"])

        # Print meta data
        if self.verbose:
            print(
                "sample_id:", sample_id,
                "target:", audio_files["target"],
                "target_position:", [round(x, 3) for x in target_position],
                "interferes:", audio_files["interferes"],
                "interference_positions:", [[round(x, 3) for x in pos] for pos in source_positions[1:]],
                "azimuth_deg:", azimuth_deg,
                "room_dim:", room_dim,
                "rt60:", round(rt60[0][0], 1),
                "snr:", snr,
                "gender:", genders,
                "file:", filename
            )

        return

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


