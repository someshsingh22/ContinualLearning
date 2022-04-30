import numpy as np
import torch


class Memory:
    def __init__(self, args, nc_per_task, compute_offsets):
        self.args = args
        self.nc_per_task = nc_per_task
        self.compute_offsets = compute_offsets

        self.n_memories = args.n_memories
        self.mem_cnt = 0
        self.memx = torch.FloatTensor(args.n_tasks, self.n_memories, *args.xdim).to(
            args.device
        )
        self.memy = torch.LongTensor(args.n_tasks, self.n_memories, *args.ydim).to(
            args.device
        )
        self.mem_feat = torch.FloatTensor(
            args.n_tasks, self.n_memories, *self.nc_per_task
        ).to(args.device)
        self.bsz = args.batch_size

    def consolidation(self, task):
        t = torch.randint(0, task, (self.bsz,))
        x = torch.randint(0, self.n_memories, (self.bsz,))
        xx = self.memx[t, x]
        feat = self.mem_feat[t, x]
        yy = self.memy[t, x]
        mask = torch.zeros(self.bsz, *self.nc_per_task)

        if self.args.offsets:
            offsets = torch.tensor([self.compute_offsets(i) for i in t])
            yy -= offsets[:, 0].to(self.args.device)

        for j in range(self.bsz):
            if self.args.offsets:
                mask[j] = torch.arange(offsets[j][0], offsets[j][1])
            else:
                mask[j] = torch.stack(
                    self.args.out_dim * [torch.arange(self.args.n_class)]
                )

        return (
            xx.to(self.args.device),
            yy.to(self.args.device),
            feat.to(self.args.device),
            mask.long().to(self.args.device),
        )

    def features_init(self, model, task):
        x = self.memx[task]
        s_ = np.s_[:]

        if self.args.offsets:
            offset1, offset2 = model.compute_offsets(task)
            x = model.VCTransform(x)
            s_ = np.s_[:, offset1:offset2]
            out = model(x, task)
        else:
            out = model(x)

        self.mem_feat[task] = torch.nn.functional.softmax(
            out[s_] / self.args.temp, dim=1
        ).data.clone()

    def update(self, x, y, task):
        endcnt = min(
            self.mem_cnt + self.args.batch_size,
            self.n_memories,
        )
        effbsz = endcnt - self.mem_cnt
        if effbsz > x.size(0):
            effbsz = x.size(0)
            endcnt = self.mem_cnt + effbsz
        self.memx[task, self.mem_cnt : endcnt].copy_(x.data[:effbsz])
        self.memy[task, self.mem_cnt : endcnt].copy_(y.data[:effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0
