import random

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms

MEAN, STD = (0.082811184, 0.22163138)


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


VCTransform = transforms.Normalize(mean=[MEAN], std=[STD])


class BarlowAugment:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(20),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.GaussianBlur((3, 3), (0.1, 2.0)),
                transforms.RandomSolarize(threshold=0.2, p=0.0),
                transforms.Normalize(mean=[MEAN], std=[STD]),
            ]
        )

        self.transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(20),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.GaussianBlur((3, 3), (0.1, 2.0)),
                transforms.RandomSolarize(threshold=0.5, p=0.0),
                transforms.Normalize(mean=[MEAN], std=[STD]),
            ]
        )

    def __call__(self, x):
        y1 = self.transform(x)
        # y1 = x
        y2 = self.transform_prime(x)
        return y1, y2


class Corrupt:
    def __init__(self, args, p=0.2):
        self.p = p
        self.device = args.device

    def __call__(self, x):
        x_ = x.clone()
        rand = torch.rand(x_.size()).to(self.device)
        x_[rand < self.p] = rand[rand < self.p].float()
        return x, x_


def one_hot(x, num_classes, on_value=1.0, off_value=0.0, device="cuda"):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(
        1, x, on_value
    )


def mixup_target(target, num_classes, lam=1.0, smoothing=0.0, device="cuda"):
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    y1 = one_hot(
        target, num_classes, on_value=on_value, off_value=off_value, device=device
    )
    y2 = one_hot(
        target.flip(0),
        num_classes,
        on_value=on_value,
        off_value=off_value,
        device=device,
    )
    return y1 * lam + y2 * (1.0 - lam)


class Mixup:
    """Mixup that applies different params to each element
    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        prob (float): probability of applying mixup or cutmix per batch or element
        num_classes (int): number of classes for target
    """

    def __init__(self, args):
        self.batch_size = args.batch_size
        self.mixup_alpha = args.mixup_alpha
        self.mix_prob = args.mixup_prob
        self.num_classes = args.n_ways

    def __call__(self, x, target):
        assert self.batch_size % 2 == 0, "Batch size should be even when using this"

        lam = np.ones(self.batch_size, dtype=np.float32)
        lam_mix = np.random.beta(
            self.mixup_alpha, self.mixup_alpha, size=self.batch_size
        )
        lam_batch = np.where(
            np.random.rand(self.batch_size) < self.mix_prob,
            lam_mix.astype(np.float32),
            lam,
        )

        x_orig = x.clone()
        for i in range(self.batch_size):
            j = self.batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.0:
                x[i] = x[i] * lam + x_orig[j] * (1 - lam)

        lam = torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)
        target = mixup_target(target, self.num_classes, lam, 0, x.device)
        return x, target
