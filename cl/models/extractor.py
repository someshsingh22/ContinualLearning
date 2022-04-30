import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNetConv1(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False,
    ):
        super(ResNetConv1, self).__init__()
        self.layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        return self.layer(x)


class VCResNet(nn.Module):
    def __init__(self, args):
        super(VCResNet, self).__init__()
        self.args = args
        self.resnet = resnet18(num_classes=args.n_class)
        self.resnet.conv1 = ResNetConv1()


class VCResNetSlow(VCResNet):
    def __init__(self, args):
        super(VCResNetSlow, self).__init__(args)

    def forward(self, x, return_feat=False):
        feats = []
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        feats.append(x.clone())
        x = self.resnet.layer2(x)
        feats.append(x.clone())
        x = self.resnet.layer3(x)
        feats.append(x.clone())
        x = self.resnet.layer4(x)
        feats.append(x.clone())

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        if return_feat:
            return feats
        else:
            return x


class VCResNetFast(VCResNet):
    def __init__(self, args):
        super(VCResNetFast, self).__init__(args)
        self.fc = nn.Linear(512, self.args.n_class)

    def forward(self, x, feats):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = x * feats[0]
        x = self.resnet.layer2(x)
        x = x * feats[1]
        x = self.resnet.layer3(x)
        x = x * feats[2]
        x = self.resnet.layer4(x)
        x = x * feats[3]

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class LSTMMarket(nn.Module):
    def __init__(self, args):
        super(LSTMMarket, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.out_dim = args.out_dim
        self.n_class = args.n_class
        self.input_dim = args.input_dim
        self.lstm = nn.LSTM(
            args.input_dim, args.hidden_dim, num_layers=1, batch_first=True, dropout=0.5
        )
        self.linear = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.relu = nn.ReLU()


class LSTMSlow(LSTMMarket):
    def __init__(self, args):
        super(LSTMSlow, self).__init__(args)
        self.args = args

    def forward(self, x, return_feat=True):
        feat = []
        h, _ = self.lstm(x)
        h = h[:, -1, :]
        feat.append(h.clone())
        h = self.relu(self.linear(h))
        feat.append(h.clone())
        if return_feat:
            return feat
        else:
            return h


class LSTMFast(LSTMMarket):
    def __init__(self, args):
        super(LSTMFast, self).__init__(args)
        self.args = args

        for i in range(self.args.out_dim):
            setattr(
                self, "fc" + str(i), nn.Linear(self.args.hidden_dim, self.args.n_class)
            )

    def forward(self, x, feat):
        h, _ = self.lstm(x)
        h = h[:, -1, :]
        h = h * feat[0]
        h = self.relu(self.linear(h))
        h = h * feat[1]
        outs = [getattr(self, "fc" + str(i))(h) for i in range(self.args.n_class)]
        return torch.stack(outs).permute(1, 0, 2)
