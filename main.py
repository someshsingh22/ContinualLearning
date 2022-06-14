# Imports
import argparse

from cl.utils import get_args

# Global Variables


# Functions


# Arguments
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.model_args, args.data_args, args.training_args = get_args(
    output_dir=args.output_dir, dataset_name=args.dataset_name
)

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
