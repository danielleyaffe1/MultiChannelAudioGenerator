import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py


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


def extract_samples_from_h5(h5_file):
    """Extract sample identifiers (target/sentence pairs) from h5 file."""
    
    videos = h5_file["videos_path"][:]
    videos = [v.decode("utf-8") for v in videos]
    print(f"Extracted {len(videos)} samples from {h5_file.filename}.")
    print(f"Example samples: {videos[0]}")
    h5_file.close()
    
    return videos


def snr_distribution_by_split(metadata_file, output_dir):
    """
    Compare SNR distribution across train, val, and test splits.
    
    Args:
        metadata_file: Path to metadata JSON file
        train_h5: Path to train.h5
        val_h5: Path to val.h5
        test_h5: Path to test.h5
        output_dir: Directory to save comparison plots
    """
    # Load metadata
    metadata = load_dataset(metadata_file)
    df = pd.DataFrame(metadata)
    
    # Extract sample identifiers from each h5 file
    train_h5 = h5py.File("/gpfs0/bgu-br/users/yaffedan/MultiChannelAudioGenerater/train.h5",'r')
    val_h5 = h5py.File("/gpfs0/bgu-br/users/yaffedan/MultiChannelAudioGenerater/val.h5",'r')
    test_h5 = h5py.File("/gpfs0/bgu-br/users/yaffedan/MultiChannelAudioGenerater/test.h5",'r')

    train_samples = set(extract_samples_from_h5(train_h5))
    val_samples = set(extract_samples_from_h5(val_h5))
    test_samples = set(extract_samples_from_h5(test_h5))

    
    # Create identifier from target and sentence (matching h5 key structure)
    df['sample_key'] = 'cleandata/' + df['target'] + '/' + df['sentence'] + '.mpg'
    
    # Assign split labels
    df['split'] = df['sample_key'].apply(lambda x: 
        'train' if x in train_samples else 
        'val' if x in val_samples else 
        'test' if x in test_samples else 
        'unknown'
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    # === 1. SNR Distribution Comparison (Boxplot) ===
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='split', y='snr', palette='Set2')
    plt.title("SNR Distribution Across Train/Val/Test Splits")
    plt.ylabel("SNR (dB)")
    plt.xlabel("Split")
    plt.savefig(os.path.join(output_dir, 'snr_distribution_by_split_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # === 2. SNR Histogram Comparison ===
    plt.figure(figsize=(12, 6))
    for split in ['train', 'val', 'test']:
        split_data = df[df['split'] == split]['snr']
        plt.hist(split_data, alpha=0.5, label=f'{split} (n={len(split_data)})', bins=len(split_data.unique()))
    plt.title("SNR Distribution Comparison")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'snr_distribution_by_split_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # === 3. SNR Value Counts per Split ===
    plt.figure(figsize=(10, 6))
    snr_split_counts = df.groupby(['split', 'snr']).size().unstack(fill_value=0)
    snr_split_counts.T.plot(kind='bar', figsize=(12, 6))
    plt.title("SNR Count per Split")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Count")
    plt.legend(title="Split")
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(output_dir, 'snr_count_per_split.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # === 4. Print Statistics ===
    print("\n=== SNR Distribution Statistics by Split ===")
    for split in ['train', 'val', 'test']:
        split_data = df[df['split'] == split]
        if len(split_data) > 0:
            print(f"\n{split.upper()}:")
            print(f"  Total samples: {len(split_data)}")
            print(f"  SNR mean: {split_data['snr'].mean():.2f} dB")
            print(f"  SNR std: {split_data['snr'].std():.2f} dB")
            print(f"  SNR range: [{split_data['snr'].min()}, {split_data['snr'].max()}]")
            print(f"  SNR value counts:\n{split_data['snr'].value_counts().sort_index()}")
        else:
            print(f"\n{split.upper()}: No samples found")
    
    print("\n=== Overall SNR Distribution ===")
    print(f"Total metadata samples: {len(df)}")
    print(f"Samples found in splits: {len(df[df['split'] != 'unknown'])}")
    print(f"Unmatched samples: {len(df[df['split'] == 'unknown'])}")
    
    return df


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
    dataset_dir = os.path.join('/gpfs0/bgu-br/users/yaffedan/MultiChannelAudioGenerater', 'Reverb_Babble', 'dataset.json')  # Replace with meta data .json file
    output_dir = "EDA_plots"

    EDA_main(dataset_dir, output_dir, simulation='room')
    # snr_distribution_by_split(dataset_dir, 
    #                           output_dir="MultiChannelAudioGenerater/EDA_data_splits")


