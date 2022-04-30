import random

from learn2learn.data import DataDescription
from learn2learn.data.transforms import TaskTransform


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
