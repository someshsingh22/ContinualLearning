# import utilities
import argparse
import logging
import time
from copy import deepcopy

# import ml libraries
import learn2learn as l2l
import torch
import torch.nn.functional as F
from learn2learn.data import TaskDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# import custom libraries
from cl.data import ContinousNWays, ImageData, MetaLoader, Mixup, MyDS
from cl.models import DualNetVC as DualNet
from cl.utils import VCMetrics, checkpoint, deterministic, load_image_data_pickle

# set logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    filename="./logs/log.txt",
    filemode="w+",
)

# default `log_dir` is "runs" - we'll be more specific here
parser = argparse.ArgumentParser(description="DualNet-Image")

parser.add_argument("--path", type=str, default="./data/image_data.pickle")
parser.add_argument(
    "--save_path",
    type=str,
    default="results/",
    help="save models at the end of training",
)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--n_runs", type=int, default=5)
parser.add_argument("--n_epochs", type=int, default=2)
parser.add_argument("--inner_steps", type=int, default=1)
parser.add_argument("--n_outer", type=int, default=1)

parser.add_argument("--n_ways", type=int, default=5)
parser.add_argument("--n_class", type=int, default=74)
parser.add_argument("--n_tasks", type=int, default=14)
parser.add_argument(
    "--n_memories", type=int, default=50, help="number of memories per task"
)
parser.add_argument("--xdim", type=tuple, default=(1, 20, 20), help="Input Dimensions")
parser.add_argument("--ydim", type=tuple, default=(), help="Input Dimensions")
parser.add_argument("--offsets", type=bool, default=True)

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--ssl_lr", type=float, default=0.001)
parser.add_argument(
    "--memory_strength",
    default=1.0,
    type=float,
    help="memory strength (meaning depends on memory)",
)
parser.add_argument("--reg", default=1.0, type=float)
parser.add_argument(
    "--temp", type=float, default=1.0, help="temperature for distilation"
)
parser.add_argument("--beta", type=float, default=0.3)
parser.add_argument("--mixup_prob", type=float, default=0.3)
parser.add_argument("--mixup_alpha", type=float, default=0.2)
parser.add_argument("--ssl_loss", type=str, default="BarlowTwins")

args = parser.parse_args()

# main code
if __name__ == "__main__":
    metrics = VCMetrics(args)
    mixup = Mixup(args)
    data = load_image_data_pickle(args.path)
    device = torch.device(args.device)  # use device specified in args
    train_data, test_data = MyDS(data.train_ds.samples, data.train_ds.labels), MyDS(
        data.test_ds.samples, data.test_ds.labels
    )
    train_data, test_data = l2l.data.MetaDataset(train_data), l2l.data.MetaDataset(
        test_data
    )
    logging.info("data loaded")

    transforms = [
        ContinousNWays(train_data, args.n_ways, args),
        l2l.data.transforms.LoadData(train_data),
    ]
    train_taskset = TaskDataset(
        train_data, transforms, num_tasks=args.n_class // args.n_ways
    )
    test_taskset = TaskDataset(
        test_data, transforms, num_tasks=args.n_class // args.n_ways
    )

    with tqdm(
        range(args.n_runs), desc="Runs Loop", leave=False, position=0, total=args.n_runs
    ) as pbar:
        for run in pbar:
            logging.info("Run {}".format(run))
            writer = SummaryWriter(f"{args.save_path}/test_VC_{time.time()}")
            deterministic(args.seed + run)

            # create model and losses
            model = DualNet(args).to(device)
            CLoss = torch.nn.CrossEntropyLoss()
            KLLoss = torch.nn.KLDivLoss()

            opt = torch.optim.SGD(model.parameters(), lr=args.lr)
            ssl_opt = torch.optim.SGD(model.SlowLearner.parameters(), lr=args.ssl_lr)

            with tqdm(
                enumerate(MetaLoader(train_taskset, args, train=True)),
                desc="Task Loop",
                total=args.n_tasks,
                leave=False,
                position=0,
            ) as outer:
                for task, train_loader in outer:
                    logging.info("Running Task {}".format(task))
                    model.train()
                    if task > 0:
                        model.memory.features_init(model, task - 1)

                    for epoch in range(args.n_epochs):
                        logging.info("Epoch {}".format(epoch))
                        with tqdm(
                            enumerate(train_loader),
                            desc="Train Loop",
                            total=len(train_loader),
                        ) as inner:
                            for i, (x, y) in inner:
                                x, y = x.to(device), y.to(device)
                                model.memory.update(x, y, task)

                                for j in range(args.n_outer):
                                    weights_before = deepcopy(model.state_dict())
                                    SSL_loss = 0
                                    for _ in range(args.inner_steps):
                                        model.zero_grad()
                                        if task > 0:
                                            (
                                                xx,
                                                yy,
                                                target,
                                                mask,
                                            ) = model.memory.consolidation(task)
                                            x1, x2 = model.barlow_augment(xx)
                                        else:
                                            x1, x2 = model.barlow_augment(x)
                                        SSLLoss = model.SlowLearner((x1, x2))
                                        SSLLoss.backward()
                                        ssl_opt.step()
                                        writer.add_scalar(
                                            "SSL loss",
                                            SSLLoss.item(),
                                            (epoch * len(train_loader) + i)
                                            * args.n_outer
                                            + j,
                                        )

                                    weights_after = model.state_dict()
                                    new_params = {
                                        name: weights_before[name]
                                        + (
                                            (weights_after[name] - weights_before[name])
                                            * args.beta
                                        )
                                        for name in weights_before.keys()
                                    }
                                    model.load_state_dict(new_params)
                                correct = 0
                                total = 0
                                for inner in range(args.inner_steps):
                                    model.zero_grad()
                                    x = model.VCTransform(x)
                                    offset1, offset2 = model.compute_offsets(task)
                                    x, y = mixup(x, y - offset1)
                                    pred = model(x, task)
                                    loss1 = CLoss(pred[:, offset1:offset2], y)
                                    writer.add_scalar(
                                        "training loss",
                                        loss1.item(),
                                        (epoch * len(train_loader) + i)
                                        * args.inner_steps
                                        + inner,
                                    )
                                    loss2, loss3 = 0, 0
                                    if task > 0:
                                        (
                                            xx,
                                            yy,
                                            target,
                                            mask,
                                        ) = model.memory.consolidation(task)
                                        xx = model.VCTransform(xx)
                                        pred = torch.gather(
                                            model(xx, task=None, fast=True), 1, mask
                                        )
                                        loss2 += CLoss(pred, yy)
                                        loss3 = args.reg * KLLoss(
                                            F.log_softmax(pred / args.temp, dim=1),
                                            target,
                                        )
                                    loss = loss1 + loss2 + loss3
                                    writer.add_scalar(
                                        "final loss",
                                        loss.item(),
                                        (epoch * len(train_loader) + i)
                                        * args.inner_steps
                                        + inner,
                                    )
                                    loss.backward()
                                    opt.step()

                    model.eval()
                    with tqdm(
                        enumerate(MetaLoader(test_taskset, args, train=False)),
                        desc="Test Loop",
                        total=task,
                        leave=False,
                        position=0,
                    ) as inner:
                        for task_t, te_loader in inner:
                            if task_t > task:
                                break
                            correct = 0
                            for data, target in te_loader:
                                data, target = data.to(device), target.to(device)
                                data = model.VCTransform(data)
                                logits = model(data, task_t)
                                # loss = CLoss(logits, target)
                                pred = logits.argmax(dim=1, keepdim=True)
                                correct += pred.eq(target.view_as(pred)).sum().item()
                            acc = correct / (len(data) * len(te_loader))
                            metrics.update_metric(run, task, task_t, acc)
                            logging.info("Task {} Acc: {:.4f}".format(task_t, acc))
            checkpoint(run, model, opt, ssl_opt, args)
    print(metrics)
    metrics.plot()
