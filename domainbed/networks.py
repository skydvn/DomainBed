# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from functools import reduce

from domainbed.lib import wide_resnet
import copy


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""

    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth'] - 2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""

    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError


def UKIE_Featurizer(input_shape, lat_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(lat_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(lat_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        return ResNet(lat_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)


class FEncoder(nn.Module):
    """ Encoder for UKIE """
    def __init__(self, n_inputs, hparams):
        super(FEncoder, self).__init__()
        self.in_channel = n_inputs[0]
        self.hid1_channel = hparams['f_hid1_channel']
        self.hid2_channel = hparams['f_hid2_channel']
        self.out_channel = hparams['f_out_channel']
        self.conv_in = nn.Sequential(
            nn.Conv2d(self.in_channel, self.hid1_channel, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.hid1_channel)
        )
        self.conv_hid1 = nn.Sequential(
            nn.Conv2d(self.hid1_channel, self.hid2_channel, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.hid2_channel),
        )
        self.conv_hid2 = nn.Sequential(
            nn.Conv2d(self.hid2_channel, self.hid2_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.hid2_channel),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.hid2_channel, self.out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.out_channel),
        )

    def forward(self, x):
        z = self.conv_in(x)
        z = self.conv_hid1(z)
        z = self.conv_hid2(z)
        z = self.conv_out(z)
        return z


class UKIEncoder(nn.Module):
    """ Encoder for UKIE """
    def __init__(self, iv_flag, hparams):
        super(UKIEncoder, self).__init__()
        self.in_channel = hparams['f_out_channel']
        self.hid_channel = hparams['iv_hid_channel']
        self.out_channel = hparams[f'{iv_flag}_out_channel']
        self.conv_in = nn.Sequential(
            nn.Conv2d(self.in_channel, self.hid_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.hid_channel)
        )
        self.conv_hid = nn.Sequential(
            nn.Conv2d(self.hid_channel, self.hid_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.hid_channel),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.hid_channel, self.out_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.out_channel),
        )

    def forward(self, x):
        z = self.conv_in(x)
        z = self.conv_hid(z)
        z = self.conv_out(z)
        return z


class Aux_Decoder(nn.Module):
    """ Encoder for UKIE """
    def __init__(self, n_outputs, hparams):
        super(Aux_Decoder, self).__init__()
        self.in_channel = int(hparams['inv_out_channel'] + hparams['var_out_channel'])
        self.hid1_channel = hparams['aux_dec_hid1_channel']
        self.hid2_channel = hparams['aux_dec_hid2_channel']
        self.out_channel = n_outputs[0]
        self.up_in = nn.Sequential(
            nn.ConvTranspose2d(self.in_channel, self.hid1_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.up_hid1 = nn.Sequential(
            nn.ConvTranspose2d(self.hid1_channel, self.hid1_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.hid1_channel),
        )
        self.up_hid2 = nn.Sequential(
            nn.ConvTranspose2d(self.hid1_channel, self.hid2_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.hid2_channel),
        )
        self.up_out = nn.Sequential(
            nn.ConvTranspose2d(self.hid2_channel, self.out_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.SELU(),
        )

    def forward(self, z):
        z = self.up_in(z)
        z = self.up_hid1(z)
        z = self.up_hid2(z)
        x = self.up_out(z)
        return x


class Aux_Classifier(nn.Module):
    """ Encoder for UKIE """
    def __init__(self, input_shape, num_classes, hparams):
        super(Aux_Classifier, self).__init__()
        self.in_channel = hparams['inv_out_channel']
        self.hid_channel = hparams['aux_cls_hid_channel']
        self.up_in = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.hid_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.up_hid = nn.Sequential(
            nn.Conv2d(in_channels=self.hid_channel, out_channels=self.hid_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.lin_shape = self.get_lin_shape(torch.zeros(input_shape))
        self.up_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(reduce(lambda x, y: x*y, self.lin_shape[1:])), num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        z = self.up_in(x)
        z = self.up_hid(z)
        z = self.up_hid(z)
        y = self.up_out(z)
        return y

    def get_lin_shape(self, x):
        z = self.up_in(x)
        z = self.up_hid(z)
        z = self.up_hid(z)
        return z.size()


class WholeUKIE(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeUKIE, self).__init__()
        self.encoder = FEncoder(input_shape, hparams)
        self.inv = UKIEncoder(iv_flag="inv", hparams=hparams)
        self.var = UKIEncoder(iv_flag="var", hparams=hparams)
        self.inv_shape, self.var_shape, self.lat_shape = self.get_shape(torch.zeros(input_shape).unsqueeze(0))
        self.aux_decoder = Aux_Decoder(input_shape, hparams)
        self.aux_classifier = Aux_Classifier(self.inv_shape, num_classes, hparams)
        self.featurizer = UKIE_Featurizer(input_shape, self.lat_shape, hparams)
        self.classifier = Classifier(
            self.featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            self.featurizer, self.classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        enc = self.encoder(x)                  # Dimension complexity reduction
        inv_enc = self.inv(enc)                # Invariant extractor
        var_enc = self.var(enc)                # Variant extractor
        lat = torch.cat((inv_enc, var_enc), dim=1)
        rec = self.aux_decoder(lat)            # Auxiliary Decoder
        logits = self.aux_classifier(inv_enc)  # Auxiliary Classifier
        return logits, rec, inv_enc, var_enc

    def predict(self, x):
        enc = self.encoder(x)  # dimension complexity reduction
        inv_enc = self.inv(enc)  # invariant extractor
        var_enc = self.var(enc)  # variant extractor
        lat = torch.cat((inv_enc, var_enc), dim=1)
        return self.net(lat)

    def get_shape(self, x):
        enc = self.encoder(x)    # dimension complexity reduction
        inv_enc = self.inv(enc)  # invariant extractor
        var_enc = self.var(enc)  # variant extractor
        lat = torch.cat((inv_enc, var_enc), dim=1)
        return inv_enc.size(), var_enc.size(), lat.size()
