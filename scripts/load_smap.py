import numpy as np
from pathlib import Path


def load_and_concat(folder: Path, channels: int = 25) -> tuple[np.ndarray, list[str]]:
    """
    Load all .npy files in a folder with a given number of channels,
    and concatenate along the time axis.
    Returns the concatenated array and the list of filenames that were included.
    """
    arrays = []
    included_files = []
    for file in sorted(folder.glob("*.npy")):
        arr = np.load(file)
        if arr.shape[1] != channels:
            print(f"Skipping {file}, channels={arr.shape[1]}")
            continue
        arrays.append(arr)
        included_files.append(file.stem)
        print(f"Loaded {file} with shape {arr.shape}")

    if not arrays:
        raise ValueError(f"No files with {channels} channels found in {folder}")

    concatenated = np.concatenate(arrays, axis=0)
    print(f"Concatenated shape: {concatenated.shape}")
    return concatenated, included_files


def main():
    # User provided root
    kaggle_root = Path("C:/Users/User/Downloads/smap/data/data")
    train_folder = kaggle_root / "train"
    test_folder = kaggle_root / "test"
    labels_folder = kaggle_root / "labels"

    # Project raw data directory
    project_root = Path(__file__).parent.parent
    raw_root = project_root / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    # Only C=25
    print("Processing train data (C=25)...")
    train_25, _ = load_and_concat(train_folder, channels=25)
    np.save(raw_root / "smap_train.npy", train_25)

    print("Processing test data (C=25)...")
    test_25, test_filenames = load_and_concat(test_folder, channels=25)
    np.save(raw_root / "smap_test.npy", test_25)

    # Labels
    labels_list = []
    if labels_folder.exists():
        print("Processing labels from labels/ folder...")
        for stem in test_filenames:
            label_file = labels_folder / f"{stem}.npy"
            if label_file.exists():
                arr = np.load(label_file)
                labels_list.append(arr.reshape(-1))
            else:
                print(f"Warning: Label file {label_file} not found.")
    else:
        csv_path = kaggle_root.parent.parent / "labeled_anomalies.csv"
        if csv_path.exists():
            print(f"Labels folder not found. Generating labels from {csv_path}...")
            import csv
            import ast

            # Load CSV into a dictionary for quick lookup
            anomaly_map = {}
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    anomaly_map[row['chan_id']] = ast.literal_eval(row['anomaly_sequences'])
            
            for stem in test_filenames:
                # Find the matching channel in the CSV
                # The stem might be 'P-1' but the CSV has 'P-1'
                if stem in anomaly_map:
                    # Get length from the loaded test file to be safe
                    # But we already have the test_25 array... 
                    # Actually, we need the length of the INDIVIDUAL file.
                    # Let's reload just the shape or use the test_filenames and match them.
                    # We can load the file again or keep track of lengths in load_and_concat.
                    arr = np.load(test_folder / f"{stem}.npy")
                    n_vals = len(arr)
                    
                    label_arr = np.zeros(n_vals, dtype=int)
                    for start, end in anomaly_map[stem]:
                        label_arr[start:end+1] = 1 # inclusive range usually
                    labels_list.append(label_arr)
                else:
                    print(f"Warning: No anomaly info found for {stem} in CSV.")
                    # If not in CSV, assume no anomalies (or we don't know the length)
                    arr = np.load(test_folder / f"{stem}.npy")
                    labels_list.append(np.zeros(len(arr), dtype=int))
        else:
            print("Error: Neither labels folder nor labeled_anomalies.csv found.")

    if labels_list:
        labels_concat = np.concatenate(labels_list, axis=0).astype(int)
        print(f"Labels shape: {labels_concat.shape}")
        np.save(raw_root / "smap_labels.npy", labels_concat)
        print(f"Saved labels to {raw_root / 'smap_labels.npy'}")

    print("Processing complete.")


if __name__ == "__main__":
    main()
