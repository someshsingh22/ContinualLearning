import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image
from PIL.ImageOps import invert
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, Dataset, TensorDataset


class MyDS(Dataset):
    def __init__(self, X, y):
        self.samples = torch.Tensor(X)
        self.labels = torch.LongTensor(y)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (self.samples[idx].view(1, 20, 20) / 255, self.labels[idx])

class Dataset(data.Dataset):
    def __init__(self,path,split):
        
        self.json_data = json.load(open(path))
        self.dataset=self.json_data[split]
        self.len=len(self.dataset)
    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.len
class ImageData:
    def __init__():
        # train_ds
        # test_ds
        # images_train
        # images_test
        # names_train
        # names_test
        # dloader
        # mapping

        pass

class ContinousNWays(TaskTransform):
    def __init__(self, dataset, n_ways, args):
        super(ContinousNWays, self).__init__(dataset)
        self.n = n_ways
        self.indices_to_labels = dict(dataset.indices_to_labels)
        self.id = 0
        self.labels = list(range(args.n_class))

    def new_task(self):
        if self.id % 14 == 0:
            self.id = 0
        task_description = []
        labels_to_indices = dict(self.dataset.labels_to_indices)
        classes = self.labels[self.id * self.n : (self.id + 1) * self.n]
        for cl in classes:
            for idx in labels_to_indices[cl]:
                task_description.append(DataDescription(idx))
        self.id += 1
        return task_description

    def __call__(self, task_description):
        return self.new_task()
class MetaLoader(object):
    def __init__(self, taskset, args, train=True):
        bs = args.batch_size if train else 64

        self.tasks = taskset
        self.loaders = []
        for X, y in self.tasks:
            
            dl = DataLoader(
                ds, batch_size=bs, shuffle=True, pin_memory=False, drop_last=True
            )
            self.loaders.append(dl)

    def __getitem__(self, idx):
        return self.loaders[idx]

    def __len__(self):
        return len(self.loaders)


class MarketTaskset:
    def __init__(self, args, maxSym=15, maxDay=15, split="Train") -> None:
        df = pd.read_csv(args.path)
        df.interpolate(axis=1, method="linear", inplace=True)
        df = df[(df["sym"] < maxSym) & (df["day"] < maxDay)]
        if split == "Train":
            df = df[df["day"] < maxDay * 0.75]
        elif split == "Test":
            df = df[df["day"] >= maxDay * 0.75]
        else:
            raise NotImplementedError
        self.df = df
        self.tasks = [
            self.normalizeTask(v) for _, v in tuple(df.groupby(["sym", "day"]))
        ]
        self.tasks = [
            self.get_scaled_pairs(X, y, args.seq_len)
            for X, y in self.tasks
            if X.size(0) > 5
        ]

    def normalizeTask(self, df):
        normalize = [
            "Open",
            "High",
            "Low",
            "Close",
            "Open_prev",
            "High_prev",
            "Low_prev",
            "Close_prev",
            "SMA_10",
            "SMA_20",
            "SMA_50",
            "SMA_200",
            "RSI_14",
            "BBL_5_2.0",
            "BBM_5_2.0",
            "BBU_5_2.0",
        ]
        try:
            df[normalize] /= df.iloc[0]["Close"]
        except Exception:
            pass

        inputs = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Dividends",
            "Open_prev",
            "High_prev",
            "Low_prev",
            "Close_prev",
            "Volume_prev",
            "Dividends_prev",
            "hurst",
            "SMA_10",
            "SMA_20",
            "SMA_50",
            "SMA_200",
            "VOL_SMA_20",
            "RSI_14",
            "BBL_5_2.0",
            "BBM_5_2.0",
            "BBU_5_2.0",
            "BBB_5_2.0",
            "BBP_5_2.0",
            "MACD_12_26_9",
            "MACDh_12_26_9",
            "MACDs_12_26_9",
            "sym",
        ]
        labels = ["(0.02, 0.01)", "(0.01, 0.005)", "(0.01, 0.02)", "(0.005, 0.01)"]
        return torch.tensor(df[inputs].values), torch.tensor(df[labels].values)

    def get_scaled_pairs(self, X, y, seq_len):
        X = normalize(X, dim=0)
        x_scaled = [X[i : i + seq_len] for i in range(len(X) - seq_len - 1)]
        y = [y[i + seq_len : i + seq_len + 1] for i in range(len(y) - seq_len - 1)]
        x_scaled = torch.stack(x_scaled)
        y = torch.stack(y)[:, 0]
        return x_scaled.float(), y.long()

    def __getitem__(self, idx):
        return self.tasks[idx]

    def __len__(self):
        return len(self.tasks)



train_dataset=Dataset('data/data_full.json','oos_train')
test_dataset=Dataset('data/data_full.json','oos_test')
train_data, test_data = l2l.data.MetaDataset(train_dataset), l2l.data.MetaDataset(
    test_dataset
)


transforms = [
    ContinousNWays(train_data,n_ways=5),
    l2l.data.transforms.LoadData(train_data),
]
train_taskset = TaskDataset(
    train_data, transforms, num_tasks=n_class // 5
)
test_taskset = TaskDataset(
    test_data, transforms, num_tasks=n_class // 5
)

# Generator function for a range
def task_gen(dataset, class_label='intent', start=0, end=150, n_ways=5):
    for i in range(start, end, n_ways):
        filter = lambda ex: ex[class_label]>= start and ex[class_label]<end
        yield dataset.filter(filter)