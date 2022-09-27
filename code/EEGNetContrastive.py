from torch import nn
import torch
from utils import getBlockOutLout, get_num_params
from einops import rearrange, repeat

import torch.nn.functional as F
from termcolor import cprint
import numpy as np


class EEGNetContrastive(nn.Module):  # Frequency domain encoder

    def __init__(self, num_chan=22, kerLen=32, drop=0.5, F1=8, D=2, F2=16, emdim=10, device='cpu'):
        super(EEGNetContrastive, self).__init__()

        # for deptwise separabel convs see https://www.youtube.com/watch?v=T7o3xvJLuHk
        # and https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d
        # in Keras SeparableConv2d is depthwise separable (i.e. depthwise + pointwise conv)
        # in Pytorch, you have to use `groups` to tell torch that you want a depthwise conv, and then you need to do
        # pointwise convs to mix the channels. This way you get depthwise separable conv

        self.num_chan = num_chan
        self.kerLen = kerLen  # half the sampling rate
        self.drop = drop
        self.F1 = F1
        self.D = D
        self.F2 = F1 * D
        self.emdim = emdim
        self.device = device

        self.fresh_model()

    def max_norm(self, eps=1e-8):
        for idx, i in enumerate(self.model):
            if idx in [2, 14]:
                if idx == 2:
                    max_val = 1.0
                elif idx == 14:
                    max_val = 0.25
                else:
                    raise
                for name, param in i.named_parameters():
                    norm = torch.linalg.norm(param)
                    desired = torch.clamp(norm, 0, max_val)
                    param = param * (desired / (eps + norm))

    def fresh_model(self):
        self.model = nn.Sequential(
            # 1x1 convs - a bit silly, but for now will do the job.
            nn.Conv2d(1, self.F1, (1, self.kerLen), padding='same', bias=False),
            nn.BatchNorm2d(self.F1, affine=True, eps=1e-05, momentum=0.9),
            # depthwise conv see https://www.youtube.com/watch?v=T7o3xvJLuHk
            nn.Conv2d(self.F1, self.F1 * self.D, (self.num_chan, 1), groups=self.F1, padding='valid',
                      bias=False),  # Depthwise conv, BUT CONSTRAIN THE NORM!!
            nn.BatchNorm2d(self.F1 * self.D, affine=True, eps=1e-05, momentum=0.9),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.drop),
            # depthwise separable conv https://github.com/seungjunlee96/Depthwise-Separable-Convolution_Pytorch
            nn.Conv2d(self.F2, self.F2, (1, 17), padding='same', groups=16, bias=False),  # depthwise conv
            nn.Conv2d(self.F2, self.F2, (1, 1), padding='same', bias=False),  # pointwise conv
            nn.BatchNorm2d(self.F2, affine=True, eps=1e-05, momentum=0.9),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.drop),
            nn.Flatten(),
            # nn.LazyLinear(out_features=self.emdim),  # max_norm constraint
            # nn.ELU(),
        ).to(self.device)

    def forward(self, x):
        x = rearrange(x, 'b c t -> b 1 c t', c=self.num_chan)
        return self.model(x)

    def train(self, x, y, lr=0.01):
        self.fresh_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        self.model.train()
        for ep in range(30):
            optimizer.zero_grad()
            pred = self.model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            self.max_norm()
        acc = np.mean([(p == l) for p, l in zip(pred.argmax(axis=1).tolist(), y.tolist())])
        return acc

    def evaluate(self, x, y):
        self.model.eval()
        pred = self.model(x)
        acc = np.mean([(p == l) for p, l in zip(pred.argmax(axis=1).tolist(), y.tolist())])
        return acc

    def summary(self, num_time_samples=120):

        cprint('*' * 90, color='blue')
        cprint(f'For {num_time_samples} time samples, the network will look like this:', color='yellow', attrs=['bold'])
        # get the network's summary for a given tensor
        if not self.model:
            self.fresh_model()
        # to init the LazyLinear layer, you need to do at least 1 fwd pass:
        o = torch.rand(size=(10, 1, self.num_chan, num_time_samples)).to(self.device)
        _ = self.model(o)

        N = 0
        for i in self.model:
            print('-' * 90)
            cprint(i, color='grey')
            print('-' * 90)
            cprint(f'Input: {o.shape}', color='green')
            for n, p in i.named_parameters():
                num_params = np.prod(p.shape)
                N += num_params
                cprint(f'{n}, {p.shape}, Nparams: {num_params}', color='blue')
            o = i(o)
            cprint(f'Output: {o.shape}', color='red')
            print()
        cprint('*' * 90, color='blue')
        cprint(f'TOTAL PARAMS: {N}', color='green', attrs=['bold'])
        cprint('*' * 90, color='blue')
        num_params


class ClassifierOnTopOfPreTrainedEEGNet(nn.Module):

    def __init__(self, emdim=0, num_classes=0):
        super(ClassifierOnTopOfPreTrainedEEGNet, self).__init__()
        assert (emdim > 0) and (num_classes > 0), "Specify the number of classes and emdim"
        self.emdim = emdim
        self.num_classes = num_classes
        self.layer0 = nn.Linear(self.emdim, self.emdim // 2)
        self.layer1 = nn.Linear(self.emdim // 2, self.num_classes)
        self.dropout = nn.Dropout(0.15)
        self.activate = nn.ELU()
        self.bn = nn.BatchNorm1d(self.emdim // 2)

        get_num_params(self)

    def forward(self, x):
        x = self.layer0(x)
        x = self.bn(x)
        x = self.activate(x)
        x = self.dropout(x)
        pred = self.layer1(x)
        return pred