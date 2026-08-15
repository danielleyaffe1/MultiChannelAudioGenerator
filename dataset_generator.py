from tools.tools import *
from tools.EDA import EDA_main
from roomaudiodatabase import MultiChannelGenerator
import json
import random
from tqdm import tqdm
import numpy as np
import os
import contextlib

PARENT_FOLDER = '/gpfs0/bgu-br/users/yaffedan/MultiChannelAudioGenerater'
AV_DATA_FOLDER = os.path.join(PARENT_FOLDER, 'cleandata')
A_DATA_FOLDER = os.path.join(PARENT_FOLDER, 'cleanaudiodata')


if __name__ == "__main__":
    # extract audio only files from .mpg files
    # extract_audio(AV_DATA_FOLDER, A_DATA_FOLDER)
    # print(f"Done extracting audio from mpg files in {AV_DATA_FOLDER}. Saved in {A_DATA_FOLDER}")

    # Define dataset parameters
    snr_values = [-10, -5, 0, 5]      # Different SNR levels
    simulation = 'room'               # Choose simulation type between 'room' or 'free_field'
    noise = 'babble'                  # Choose noise type between 'interfere' or 'babble'
    sample_rate = 16000               # Sample rate for audio files
    num_pairs_per_spk = None          # None = create pairs for all target files (max possible)
    max_total_samples = None          # Maximum data samples, None = create all possible
    save_multichannel = True          # Save multi channel mixed audio and clean seperately
    save_noise_audio = True           # Save multi channel noise audio - for IRM calculaiton...
    save_direct_audio = True          # Save multi channel direct target audio - for STOI calculaiton...
    save_binaural = False
    verbose = False
    EDA = True

    if simulation == 'room':
        room_sizes = [[10.0, 8.0, 3.0], [12.0, 9.0, 3.0], [14.0, 10.0, 3.0]]
        rt60_values = [0.2, 0.3, 0.5, 0.6]           # Different reverberation times
        sim_name = 'Reverb'
    elif simulation == 'free_field':
        room_sizes = [[12.0, 9.0, 3.0]]
        rt60_values = [0.2]                         # Unused for free field, but required for function call
        sim_name = 'FreeField'
    else:
        raise ValueError('Invalid simulation request. Please choose a simulation type: room or free_field')

    if noise == 'interfere':
        num_interfering_sources = 1
        noise_name = str(num_interfering_sources)+'Inter'
    elif noise == 'babble':
        num_interfering_sources = 3  # Number of interferes, choose >=3 for babble noise
        noise_name = 'Babble'
    else:
        raise ValueError('Invalid noise request. Please choose a noise type: interfere or babble')

    output_dir = os.path.join(PARENT_FOLDER, sim_name+'_'+noise_name, 'dataset')
    clean_output_dir = output_dir+'_clean'
    noise_output_dir = output_dir+'_noise'
    direct_output_dir = output_dir+'_direct' # For direct (target only) path audio



    # Create list of audio pairs/ combination for simulations. If a list exist reload it.
    if not os.path.exists(os.path.join(PARENT_FOLDER, "speaker_pairs.json")):
        speaker_pairs = generate_speaker_pairs(datapath=A_DATA_FOLDER, num_samples=num_pairs_per_spk, num_interferers=num_interfering_sources, max_total_samples=max_total_samples)
        if os.path.isfile(f"{output_dir}.json"):
            # Prevent appending to existing file new dataset generation!
            os.remove(f"{output_dir}.json")
            print(f"Removed existing metadata file: {output_dir}.json")
    else:
        with open(os.path.join(PARENT_FOLDER, "speaker_pairs.json"), "r") as f:
            speaker_pairs = json.load(f)
        print(f"Loaded existing speaker pairs from {os.path.join(PARENT_FOLDER, 'speaker_pairs.json')}. {len(speaker_pairs)} existing pairs")

        sample_ids = set()  # use a set for quick membership checks

        # Count existing samples in metadata file to avoid regenerating them
        line_count = 0
        if os.path.isfile(f"{output_dir}.json"):
            with open(f"{output_dir}.json", "r") as f:
                for line in f:
                    if line.strip():  # skip empty lines
                        entry = json.loads(line)
                        sample_ids.add(entry["sample_id"])
                        line_count += 1
            print(f"Existing metadata file: {output_dir}.json found and will be used/updated.")
            print(f"Found {line_count} existing samples in metadata. Will skip generating these samples again.")

    dataset_generator = MultiChannelGenerator(sample_rate=sample_rate,
                                              output_dir=output_dir,
                                              clean_output_dir=clean_output_dir,
                                              noise_output_dir=noise_output_dir,
                                              direct_output_dir=direct_output_dir,
                                              simulation=simulation,
                                              verbose=verbose)

    # Keep only samples that are NOT already in metadata in filtered_indices. If no samples exist filtered_indices is all. 
    if len(sample_ids) != 0:
        print(f"Filtering out existing samples from speaker pairs based on metadata. Existing sample count: {len(sample_ids)}")
        filtered_indices = [i for i in range(len(speaker_pairs))
                            if speaker_pairs[i]["ID"] not in sample_ids]
        print(f"{len(filtered_indices)} samples to generate after filtering out existing metadata samples. ")
        if len(speaker_pairs) - len(filtered_indices) != line_count:
            raise ValueError(f"Mismatch in counting existing samples. Expected {line_count} but found {len(speaker_pairs) - len(filtered_indices)}. Please check the metadata file and speaker pairs consistency.")
    
        random.shuffle(filtered_indices)

    else:
        filtered_indices = list(range(len(speaker_pairs)))
        random.shuffle(filtered_indices)

    idx = 0
    metadata = []
    print(f"Starting dataset generation with simulation: {simulation}, noise: {noise}, SNR values: {snr_values}, room sizes: {room_sizes}, RT60 values: {rt60_values}\n")

    with tqdm(total=len(filtered_indices), desc="Generating dataset", unit="sample") as pbar:
        while idx < len(filtered_indices):
            snr, room_dim, rt60_tgt = random.choice(snr_values), random.choice(room_sizes), random.choice(rt60_values)

            audio_files = speaker_pairs[filtered_indices[idx]]

            if idx==len(filtered_indices)-1:
                dataset_generator.verbose = True
                dataset_generator.verbose_outpur_dir = output_dir+'_plots'
                os.makedirs(output_dir+'_plots', exist_ok=True)
            try:
                sample_meta_data = dataset_generator.generate_multichannel_audio(room_dim=room_dim,
                                                                                    rt60_tgt=rt60_tgt,
                                                                                    snr=snr,
                                                                                    audio_files=audio_files,
                                                                                    num_interfering_sources=num_interfering_sources,
                                                                                    with_DRR=True,
                                                                                    save_audio=save_multichannel,
                                                                                    save_noise_audio=save_noise_audio,
                                                                                    save_direct_audio=save_direct_audio)
                if save_binaural:
                    dataset_generator.generate_binaural_audio(room_dim, rt60_tgt, snr, audio_files, num_interfering_sources,
                                                        azimuth_deg=90.0, save_audio=save_binaural)
                if save_multichannel:
                    with open(f"{output_dir}.json", "a") as f:
                        json.dump(sample_meta_data, f)  # Dump each dictionary separately
                        f.write("\n")  # Add a newline after each JSON object
                if verbose:
                    print(sample_meta_data + '\n')

                metadata.append(sample_meta_data)
                
            
            except Exception as e:
                print("\n{ Error while processing: " + str(audio_files) + '\n' + str(e) + '}\n')

            idx += 1
            pbar.update(1)
                

    print(f"Dataset generation complete! {idx} samples successfully processed. Save status: Multi channel:{save_multichannel}, Multi Channel Noise:{save_noise_audio}, Binaural:{save_binaural}")

    # EDA
    if EDA:
        EDA_main(f"{output_dir}.json", f"{output_dir}_plots", simulation=simulation)
