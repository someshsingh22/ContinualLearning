import math

import torch
from torch import nn

from dn.data import BarlowAugment, Corrupt, VCTransform
from dn.models.extractor import LSTMFast, LSTMSlow, VCResNetFast, VCResNetSlow
from dn.models.memory import Memory


class DualNetVC(torch.nn.Module):
    """
    Takes Slow, fast learners and implements dualnet
    """

    def __init__(self, args):
        # setup network
        super(DualNetVC, self).__init__()
        self.args = args
        self.n_class = args.n_class
        self.nc_per_task = int(args.n_class // args.n_tasks)

        # setup memories
        self.memory = Memory(args, (self.nc_per_task,), self.compute_offsets)

        # setup learners
        self.SlowLearner = SlowLearner(args, VCResNetSlow)
        self.FastLearner = FastLearner(args, VCResNetFast)

        # Transforms
        self.VCTransform = VCTransform
        self.barlow_augment = BarlowAugment()

    def compute_offsets(self, task):
        return self.nc_per_task * task, self.nc_per_task * (task + 1)

    def forward(self, img, task, fast=False) -> torch.Tensor:
        """
        Fast Learner Inference
        """
        feat = self.SlowLearner(img, return_feat=True)
        out = self.FastLearner(img, feat)
        if fast:
            return out

        offset1, offset2 = self.compute_offsets(task)
        if offset1 > 0:
            out[:, :offset1].data.fill_(-10e10)
        if offset2 < self.n_class:
            out[:, int(offset2) : self.n_class].data.fill_(-10e10)

        return out


class DualNetMarket(torch.nn.Module):
    """
    Takes Slow, fast learners and implements dualnet
    """

    def __init__(self, args):
        # setup network
        super(DualNetMarket, self).__init__()
        self.args = args
        self.n_class = args.n_class

        # setup memories
        self.memory = Memory(args, (args.out_dim, args.n_class), None)

        # setup learners
        self.SlowLearner = SlowLearner(args, LSTMSlow)
        self.FastLearner = FastLearner(args, LSTMFast)

        # Transforms
        self.barlow_augment = Corrupt(self.args)

    def forward(self, stock) -> torch.Tensor:
        """
        Fast Learner Inference
        """
        feat = self.SlowLearner(stock, return_feat=True)
        out = self.FastLearner(stock, feat)
        return out


class FastLearner(torch.nn.Module):
    """
    Fast Learner Takes image input and returns meta task output
    """

    def __init__(self, args, embedder):
        super(FastLearner, self).__init__()
        self.args = args
        self.embedder = embedder(args)

    def forward(self, img, feat) -> torch.Tensor:
        """
        Obtain representation from slow learner
        """
        out = self.embedder(img, feat)
        return out


class SlowLearner(torch.nn.Module):
    """
    Slow Learner Takes two images input and returns representation
    """

    def __init__(self, args, embedder):
        super(SlowLearner, self).__init__()
        self.args = args
        self.embedder = embedder(args)

    def forward(self, input, return_feat=False):
        """
        Obtain representation from slow learner
        """
        if return_feat:
            feat = self.embedder(input, return_feat=True)
            return feat

        else:
            emb, emb_ = self.embedder(input[0], return_feat=False), self.embedder(
                input[1], return_feat=False
            )
            emb = (emb - emb.mean(0)) / emb.std(0)
            emb_ = (emb_ - emb_.mean(0)) / emb_.std(0)

            if self.args.ssl_loss == "BarlowTwins":
                return self.barlow_twins_losser(emb, emb_)
            elif self.args.ssl_loss == "SimCLR":
                return self.SimCLR_losser(emb, emb_)
            else:
                raise NotImplementedError

    def barlow_twins_losser(self, z1, z2):
        """
        Input: z1, z2 embeddings
        Returns loss by barlo twins method
        """
        N, D = z1.size(0), z1.size(1)
        c_ = torch.mm(z1.T, z2) / N
        diag = torch.eye(D).to(self.args.device)
        c_diff = (c_ - diag).pow(2)
        c_diff[~torch.eye(D, dtype=bool)] *= 2e-3
        loss = c_diff.sum()
        return loss

    def SimCLR_losser(self, z1, z2, temp=100, eps=1e-6):
        out = torch.cat([z1, z2], dim=0)
        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / temp)
        neg = sim.sum(dim=1)

        row_sub = torch.Tensor(neg.shape).fill_(math.e ** (1 / temp)).cuda()
        neg = torch.clamp(neg - row_sub, min=eps)
        pos = torch.exp(torch.sum(z1 * z2, dim=-1) / temp)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()
        return loss
