from tools import *
from EDA import *
from roomaudiodatabase import MultiChannelGenerator
import json
import random
from tqdm import tqdm
import numpy as np
import math

AV_DATA_FOLDER = 'cleandata'
A_DATA_FOLDER = 'cleanaudiodata'


if __name__ == "__main__":
    # extract audio only files from .mpg files
    # extract_audio(AV_DATA_FOLDER, A_DATA_FOLDER)
    # print(f"Done extracting audio from mpg files in {AV_DATA_FOLDER}. Saved in {A_DATA_FOLDER}")

    # Define dataset parameters
    room_sizes = [[7.0, 8.0, 3.0], [10.0, 8.0, 3.0], [12.0, 9.0, 3.0]]
    rt60_values = [0.2]#[0.3, 0.5, 0.7, 0.8]  # Different reverberation times
    snr_values = [5, 10] #[-15, -10, -5, 0, 5]    #Different SNR levels
    num_interfering_sources = 1
    num_settings = len(room_sizes) * len(rt60_values) * len(snr_values)

    num_pairs_per_spk = 100

    output_dir = "NoReverv_1Inter"  # FIXME: "dataset_multichannel_audio"
    clean_output_dir = output_dir + "_clean"    # FIXME: "dataset_multichannel_audio_clean"
    save = True
    verbose = False

    # Create dataset
    speaker_pairs = generate_speaker_pairs(datapath=A_DATA_FOLDER, num_samples=num_pairs_per_spk, num_interferers=num_interfering_sources)
    dataset_generator = MultiChannelGenerator(sample_rate=16000, output_dir=output_dir, clean_output_dir=clean_output_dir, verbose=verbose, save_audio=save)

    # num_samples_per_setting = math.ceil(len(speaker_pairs)/num_settings)
    #
    # if num_samples_per_setting*num_settings > len(speaker_pairs):
    #     raise ValueError('Need to generate more pairs per speaker using generate_speaker_pairs function (select higher value for "num_samples")')

    indexes = list(range(len(speaker_pairs)))
    random.shuffle(indexes)

    idx = 0
    metadata = []
    with tqdm(total=len(speaker_pairs), desc="Generating dataset", unit="sample") as pbar:
        while idx < len(speaker_pairs):
            snr, room_dim, rt60_tgt = random.choice(snr_values), random.choice(room_sizes), random.choice(rt60_values)

            audio_files = speaker_pairs[indexes[idx]]
            sample_meta_data = dataset_generator.generate_multichannel_audio(room_dim, rt60_tgt, snr,
                                                                             audio_files,
                                                                             num_interfering_sources)
            if save:
                with open(f"{output_dir}.json", "a") as f:
                    json.dump(sample_meta_data, f)  # Dump each dictionary separately
                    f.write("\n")  # Add a newline after each JSON object
            if verbose:
                print(sample_meta_data)
            metadata.append(sample_meta_data)
            pbar.update(1)
            idx += 1

    print(f"Dataset generation complete! {idx} samples processed. Save status: {save}.")

    # EDA
    if save:
        main(f"{output_dir}.json",f"{output_dir}_plots")







