from torch import nn
import torch
from utils import getBlockOutLout
from einops import rearrange, repeat

import torch.nn.functional as F
from termcolor import cprint
import numpy as np


def get_num_params(model):
    # get number of parameters
    a = 0
    for p in model.parameters():
        a += p.flatten().shape[0]
    print(f'number of parameters in model: {a}')


class TFC_new(nn.Module):  # Frequency domain encoder

    def __init__(self, configs):
        super(TFC_new, self).__init__()

        # compute the right size (useful when you change configs)
        Lout1 = getBlockOutLout(Lin=configs.TSlength_aligned,
                                padding=configs.kernel_size // 2,
                                dilation=1,
                                kernel_sz=configs.kernel_size,
                                stride=configs.stride)
        Lout2 = getBlockOutLout(Lin=Lout1, padding=4, dilation=1, kernel_sz=8, stride=1)
        Lout3 = getBlockOutLout(Lin=Lout2, padding=4, dilation=1, kernel_sz=8, stride=1)

        self.conv_block1_t = nn.Sequential(
            nn.Conv1d(configs.input_channels,
                      32,
                      kernel_size=configs.kernel_size,
                      stride=configs.stride,
                      bias=False,
                      padding=(configs.kernel_size // 2)), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1), nn.Dropout(configs.dropout))

        self.conv_block2_t = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
                                           nn.BatchNorm1d(64), nn.ReLU(),
                                           nn.MaxPool1d(kernel_size=2, stride=2, padding=1))

        self.conv_block3_t = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.projector_t = nn.Sequential(nn.Linear(Lout3 * configs.final_out_channels, 256), nn.BatchNorm1d(256),
                                         nn.ReLU(), nn.Linear(256, 128))

        self.conv_block1_f = nn.Sequential(
            nn.Conv1d(configs.input_channels,
                      32,
                      kernel_size=configs.kernel_size,
                      stride=configs.stride,
                      bias=False,
                      padding=(configs.kernel_size // 2)), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1), nn.Dropout(configs.dropout))

        self.conv_block2_f = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
                                           nn.BatchNorm1d(64), nn.ReLU(),
                                           nn.MaxPool1d(kernel_size=2, stride=2, padding=1))

        self.conv_block3_f = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.projector_f = nn.Sequential(nn.Linear(Lout3 * configs.final_out_channels, 256), nn.BatchNorm1d(256),
                                         nn.ReLU(), nn.Linear(256, 128))

    def forward(self, x_in_t, x_in_f):
        """Time-based Contrastive Encoder"""
        x = self.conv_block1_t(x_in_t)
        x = self.conv_block2_t(x)
        x = self.conv_block3_t(x)
        h_time = x.reshape(x.shape[0], -1)
        """Cross-space projector"""
        z_time = self.projector_t(h_time)
        """Frequency-based contrastive encoder"""
        f = self.conv_block1_f(x_in_f)
        f = self.conv_block2_f(f)
        f = self.conv_block3_f(f)
        h_freq = f.reshape(f.shape[0], -1)
        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq


class TFC(nn.Module):  # Frequency domain encoder

    def __init__(self, configs):
        super(TFC, self).__init__()

        self.conv_block1_t = nn.Sequential(
            nn.Conv1d(configs.input_channels,
                      32,
                      kernel_size=configs.kernel_size,
                      stride=configs.stride,
                      bias=False,
                      padding=(configs.kernel_size // 2)), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1), nn.Dropout(configs.dropout))

        self.conv_block2_t = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
                                           nn.BatchNorm1d(64), nn.ReLU(),
                                           nn.MaxPool1d(kernel_size=2, stride=2, padding=1))

        self.conv_block3_t = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.projector_t = nn.Sequential(nn.Linear(configs.CNNoutput_channel * configs.final_out_channels, 256),
                                         nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 128))

        self.conv_block1_f = nn.Sequential(
            nn.Conv1d(configs.input_channels,
                      32,
                      kernel_size=configs.kernel_size,
                      stride=configs.stride,
                      bias=False,
                      padding=(configs.kernel_size // 2)), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1), nn.Dropout(configs.dropout))

        self.conv_block2_f = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
                                           nn.BatchNorm1d(64), nn.ReLU(),
                                           nn.MaxPool1d(kernel_size=2, stride=2, padding=1))

        self.conv_block3_f = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.projector_f = nn.Sequential(nn.Linear(configs.CNNoutput_channel * configs.final_out_channels, 256),
                                         nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 128))

    def forward(self, x_in_t, x_in_f):
        """Time-based Contrastive Encoder"""
        x = self.conv_block1_t(x_in_t)
        x = self.conv_block2_t(x)
        x = self.conv_block3_t(x)
        h_time = x.reshape(x.shape[0], -1)
        """Cross-space projector"""
        z_time = self.projector_t(h_time)
        """Frequency-based contrastive encoder"""
        f = self.conv_block1_f(x_in_f)
        f = self.conv_block2_f(f)
        f = self.conv_block3_f(f)
        h_freq = f.reshape(f.shape[0], -1)
        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq


class target_classifier(nn.Module):  # Frequency domain encoder

    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(2 * 128, 64)
        self.logits_simple = nn.Linear(64, configs.num_classes_target)

    def forward(self, emb):
        # """2-layer MLP"""
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred


class TFCmultiToOne(nn.Module):  # Frequency domain encoder

    def __init__(self, configs):
        super(TFCmultiToOne, self).__init__()

        # compute the right size (useful when you change configs)
        Lout1 = getBlockOutLout(Lin=configs.TSlength_aligned,
                                padding=configs.kernel_size // 2,
                                dilation=1,
                                kernel_sz=configs.kernel_size,
                                stride=configs.stride)
        Lout2 = getBlockOutLout(Lin=Lout1, padding=4, dilation=1, kernel_sz=8, stride=1)
        Lout3 = getBlockOutLout(Lin=Lout2, padding=4, dilation=1, kernel_sz=8, stride=1)
        self.h_dim = Lout3 * configs.final_out_channels

        self.conv_block1_t = nn.Sequential(
            nn.Conv1d(configs.input_channels,
                      32,
                      kernel_size=configs.kernel_size,
                      stride=configs.stride,
                      bias=False,
                      padding=(configs.kernel_size // 2)), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1), nn.Dropout(configs.dropout))

        self.conv_block2_t = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
                                           nn.BatchNorm1d(64), nn.ReLU(),
                                           nn.MaxPool1d(kernel_size=2, stride=2, padding=1))

        self.conv_block3_t = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.projector_t = nn.Sequential(
            nn.Linear(Lout3 * configs.final_out_channels, configs.z_dim * 2),
            nn.BatchNorm1d(configs.z_dim * 2),
            nn.ReLU(),
            nn.Linear(configs.z_dim * 2, configs.z_dim),
        )

        self.conv_block1_f = nn.Sequential(
            nn.Conv1d(configs.input_channels,
                      32,
                      kernel_size=configs.kernel_size,
                      stride=configs.stride,
                      bias=False,
                      padding=(configs.kernel_size // 2)), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1), nn.Dropout(configs.dropout))

        self.conv_block2_f = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
                                           nn.BatchNorm1d(64), nn.ReLU(),
                                           nn.MaxPool1d(kernel_size=2, stride=2, padding=1))

        self.conv_block3_f = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.projector_f = nn.Sequential(
            nn.Linear(Lout3 * configs.final_out_channels, configs.z_dim * 2),
            nn.BatchNorm1d(configs.z_dim * 2),
            nn.ReLU(),
            nn.Linear(configs.z_dim * 2, configs.z_dim),
        )
        get_num_params(self)

    def forward(self, x_in_t, x_in_f):
        """Time-based Contrastive Encoder"""
        # no matter how many channels the input has, we move them to the batch dim
        assert x_in_t.shape == x_in_f.shape, "TD and FD tensors must have the same shape."
        c = x_in_t.shape[1]

        x_in_t = rearrange(x_in_t, 'b c t -> (b c) 1 t')
        x_in_f = rearrange(x_in_f, 'b c t -> (b c) 1 t')

        x = self.conv_block1_t(x_in_t)
        x = self.conv_block2_t(x)
        x = self.conv_block3_t(x)
        h_time = x.reshape(x.shape[0], -1)
        """Cross-space projector"""
        z_time = self.projector_t(h_time)
        """Frequency-based contrastive encoder"""
        f = self.conv_block1_f(x_in_f)
        f = self.conv_block2_f(f)
        f = self.conv_block3_f(f)
        h_freq = f.reshape(f.shape[0], -1)
        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq


class ClassifierMultiToOne(nn.Module):

    def __init__(self, configs, num_chans=1):
        super(ClassifierMultiToOne, self).__init__()
        self.num_chans = num_chans
        self.logits = nn.Linear(
            configs.z_dim * self.num_chans * 2,
            configs.z_dim * self.num_chans // 2,
        )
        self.logits_simple = nn.Linear(
            configs.z_dim * self.num_chans // 2,
            configs.num_classes_target,
        )
        self.dropout = nn.Dropout(configs.dropout)
        self.SiLU = torch.nn.SiLU()
        get_num_params(self)

    def forward(self, emb):
        emb = rearrange(emb, '(b c) z -> b (c z)', c=self.num_chans)  # flatten dims (exc. batch dim)
        # rename ....
        emb = self.logits(emb)
        emb = self.dropout(emb)
        emb = self.SiLU(emb)  # try sigmoid/ others
        pred = self.logits_simple(emb)
        return pred


class ClassifierMultiToOne_spatial(nn.Module):

    def __init__(self, configs, num_chans=1):
        super(ClassifierMultiToOne_spatial, self).__init__()
        self.num_chans = num_chans
        self.logits = nn.Linear(
            configs.z_dim * 2,
            configs.z_dim // 2,
        )
        self.logits_simple = nn.Linear(
            configs.z_dim // 2,
            configs.num_classes_target,
        )

        self.spatial_filter = nn.Linear(self.num_chans, 1, bias=False)
        with torch.no_grad():
            self.spatial_filter.weight.fill_(1.0)

        get_num_params(self)

    def forward(self, emb):
        # print(emb.shape)

        emb = rearrange(emb, '(b c) t -> b t c', c=self.num_chans)
        # print(emb.shape)

        emb = self.spatial_filter(emb)
        # print(emb.shape)

        emb = rearrange(emb, 'b t 1 -> b t')
        # print(emb.shape)

        emb = torch.sigmoid(self.logits(emb))
        # print(emb.shape)

        pred = self.logits_simple(emb)
        # print(pred.shape)

        return pred


class TorchEEGNet(nn.Module):

    def __init__(self, num_chan=22, kerLen=32, drop=0.5, F1=8, D=2, F2=16, nclass=4, device='cpu'):
        super(TorchEEGNet, self).__init__()
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
        self.num_classes = nclass
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
            nn.LazyLinear(out_features=self.num_classes)  # max_norm constraint
        ).to(self.device)

    def forward(self, x):
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


class ClassifierMultiToOneEEGNetLike(nn.Module):

    def __init__(self, num_chan=7, kerLen=1, drop=0.5, F1=8, D=2, F2=16, nclass=4, device='cpu'):
        super(ClassifierMultiToOneEEGNetLike, self).__init__()
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
        self.num_classes = nclass
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
            nn.LazyLinear(out_features=self.num_classes)  # max_norm constraint
        ).to(self.device)

    def forward(self, x):
        x = rearrange(x, '(b c) t -> b 1 c t', c=self.num_chan)
        return self.model(x)

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