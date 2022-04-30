import random

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.nn.functional import one_hot
from torchvision import transforms

MEAN, STD = (0.082811184, 0.22163138)


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


VCTransform = transforms.Compose(
    [
        transforms.Resize((84, 84)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


class BarlowAugment:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(84),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(84),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
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


def mixup_target(target, num_classes, lam=1.0, device="cuda"):
    y1 = one_hot(target, num_classes).to(device)
    y2 = one_hot(target.flip(0), num_classes).to(device)
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
        target = mixup_target(target, self.num_classes, lam, x.device)
        return x, target
