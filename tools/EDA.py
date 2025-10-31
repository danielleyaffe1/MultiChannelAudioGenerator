import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Function to load the dataset (JSON files)
def load_dataset(metadata_file):
    metadata = []
    if metadata_file.endswith(".json"):
        with open(metadata_file, 'r') as f:
            for line in f:
                metadata.append(json.loads(line))  # Each line is a new JSON object
    else:
        raise ValueError(f"Non-JSON file found: {metadata_file}. Please provide a .json file.")

    return metadata


# 1. Gender distribution (based on target file path or speaker ID)
def gender_distribution(df, output_dir):
    gender_dict = {'male': 0, 'female': 0}
    male_id = ['s1', 's2', 's3', 's5', 's6', 's8', 's9', 's10', 's12', 's13', 's14', 's17', 's19', 's26', 's27', 's28', 's30', 's32']
    # Simple method: Check the speaker ID in the filename (e.g., s1 for male, s34 for female, etc.)
    for target in df['target']:
        speaker_id = target  # Assuming the file path is structured like "s1/..."
        if speaker_id in male_id:
            gender_dict['male'] += 1
        else:
            gender_dict['female'] += 1

    gender_labels = list(gender_dict.keys())
    gender_counts = list(gender_dict.values())

    # Plot gender distribution as bar plot
    plt.figure()
    plt.bar(gender_labels, gender_counts, color=['blue', 'pink'])
    plt.title("Gender Distribution")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir, 'gender_distribution.png'), dpi=300, bbox_inches='tight')  # Adjust dpi for quality and bbox for spacing
    # plt.show()


# 2. Room size distribution
def room_size_distribution(df, output_dir):
    room_sizes = df['room_dim'].apply(lambda x: tuple(x))  # Convert list to tuple for easier comparison
    room_size_counts = room_sizes.value_counts().sort_index()

    # Plot room size distribution as bar plot
    plt.figure(figsize=(8, 5))
    room_size_counts.plot(kind='bar')
    plt.title("Room Size Distribution")
    plt.xlabel("Room Dimensions (LxWxH)")
    plt.ylabel("Count")
    plt.xticks(rotation=0, ha='center')
    plt.savefig(os.path.join(output_dir, 'room_size_distribution.png'), dpi=300, bbox_inches='tight')  # Adjust dpi for quality and bbox for spacing
    # plt.show()


# 3. RT60 distribution (Box plot instead of histogram)
def rt60_distribution(df, output_dir):
    plt.figure(figsize=(8, 5))
    rt60_counts = df['rt60'].value_counts().sort_index()

    rt60_counts.plot(kind='bar', color='green')
    plt.title("Reverberation Time (RT60) Distribution")
    plt.xlabel("RT60")
    plt.ylabel("Count")
    plt.xticks(rotation=0, ha='center')
    plt.savefig(os.path.join(output_dir,'rt60_distribution.png'), dpi=300, bbox_inches='tight')  # Adjust dpi for quality and bbox for spacing
    # plt.show()


# 4. SNR Distribution (Box plot instead of histogram)
def snr_distribution(df, output_dir):
    plt.figure(figsize=(8, 5))
    snr_counts = df['snr'].value_counts().sort_index()

    snr_counts.plot(kind='bar')
    plt.title("Reverberation Time (RT60) Distribution")
    plt.xlabel("SNR")
    plt.ylabel("Count")
    plt.xticks(rotation=0, ha='center')
    plt.savefig(os.path.join(output_dir,'snr_distribution.png'), dpi=300, bbox_inches='tight')  # Adjust dpi for quality and bbox for spacing
    # plt.show()


# 5. Number of samples per speaker (Bar plot)
def samples_per_speaker(df,output_dir):
    speaker_ids = df['target']  # Extract speaker ID from file path
    speaker_counts = speaker_ids.value_counts()

    # Plot number of samples per speaker as bar plot
    plt.figure(figsize=(10, 6))
    speaker_counts.plot(kind='bar', color='purple')
    plt.title("Number of Samples Per Speaker")
    plt.xlabel("Speaker ID")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(output_dir, 'samples_per_speaker.png'), dpi=300, bbox_inches='tight')  # Adjust dpi for quality and bbox for spacing
    # plt.show()


def samples_by_gender_comb(df, output_dir):
    expected_categories = ['male', 'female', 'mixed']
    plt.figure(figsize=(8, 5))

    gender_counts = df['gender'].value_counts().reindex(expected_categories, fill_value=0)# Get counts and reindex to include missing categories

    gender_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Sample Count by Gender Combination")
    plt.xlabel("Gender Composition")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=0, ha='center')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'gender_combination_distribution.png'), dpi=300, bbox_inches='tight')
    # plt.show()


def EDA_main(dataset_dir, output_dir, simulation):
    metadata = load_dataset(dataset_dir)

    # Convert metadata to pandas DataFrame
    df = pd.DataFrame(metadata)

    os.makedirs(output_dir, exist_ok=True)

    # Run the functions
    gender_distribution(df, output_dir)
    if simulation=='room':
        room_size_distribution(df, output_dir)
        rt60_distribution(df, output_dir)
    snr_distribution(df, output_dir)
    samples_per_speaker(df, output_dir)
    samples_by_gender_comb(df, output_dir)


if __name__ == '__main__':
    # Load your dataset
    dataset_dir = "MultichannelAudio.json"  # Replace with meta data .json file
    output_dir = "EDA_plots"

    EDA_main(dataset_dir, output_dir, simulation='room')


