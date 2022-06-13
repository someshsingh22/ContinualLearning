# Imports
import argparse
import random

import learn2learn as l2l

# Global Variables


# Functions


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="./data/")
args = parser.parse_args()

# Main
if __name__ == "__main__":
    # Import AGNews Dataset
    train_dataset = Dataset("data/data_full.json", "oos_train")
    test_dataset = Dataset("data/data_full.json", "oos_test")
    train_data, test_data = l2l.data.MetaDataset(train_dataset), l2l.data.MetaDataset(
        test_dataset
    )

    transforms = [
        ContinousNWays(train_data, n_ways=5),
        l2l.data.transforms.LoadData(train_data),
    ]
    train_taskset = TaskDataset(train_data, transforms, num_tasks=n_class // 5)
    test_taskset = TaskDataset(test_data, transforms, num_tasks=n_class // 5)

    # Implement Model

    # Implemnet Training Loop

    pass
