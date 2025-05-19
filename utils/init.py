from net import *
from net.losses import ExclusionLoss, StdLoss, YIQGNGCLoss, GradientLoss, ExtendedL1Loss, GrayLoss, GrayscaleLoss, L_TV, L_spa, L_exp, L_color
from net.perLoss import PerceptualLoss
from net.noise import get_noise, NoiseNet
from utils.image_io import *
from net.downsampler import *
from net.fcn import *
from net.RRDNet import RRDNet
import os
import torch.nn as nn
    
def _init_nets(self):
        pad = 'reflection'
        left_net = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32],
            num_channels_up=[8, 16, 32],
            num_channels_skip=[0, 0, 0],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.left_net = left_net.type(torch.cuda.FloatTensor)

        right_net = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32],
            num_channels_up=[8, 16, 32],
            num_channels_skip=[0, 0, 0],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.right_net = right_net.type(torch.cuda.FloatTensor)
        self.RRD_net = RRDNet().type(torch.cuda.FloatTensor)

        kernel_net = fcn(200, 15 * 15)
        self.kernel_net = kernel_net.type(torch.cuda.FloatTensor)

def _init_images(self):
        self.images = get_imresize_downsampled(self.image, downsampling_factor=self.downsampling_factor,
                                               downsampling_number=self.downsampling_number)
        self.images_torch = [np_to_torch(image).type(torch.cuda.FloatTensor) for image in self.images]
        if self.bg_hint is not None:
            assert self.bg_hint.shape[1:] == self.image.shape[1:], (self.bg_hint.shape[1:], self.image.shape[1:])
            self.bg_hints = get_imresize_downsampled(self.bg_hint, downsampling_factor=self.downsampling_factor,
                                                     downsampling_number=self.downsampling_number)
            self.bg_hints_torch = [np_to_torch(bg_hint).type(torch.cuda.FloatTensor) for bg_hint in self.bg_hints]
        else:
            self.bg_hints = None
        if self.fg_hint is not None:
            assert self.fg_hint.shape[1:] == self.image.shape[1:]
            self.fg_hints = get_imresize_downsampled(self.fg_hint, downsampling_factor=self.downsampling_factor,
                                                     downsampling_number=self.downsampling_number)
            self.fg_hints_torch = [np_to_torch(fg_hint).type(torch.cuda.FloatTensor) for fg_hint in self.fg_hints]
        else:
            self.fg_hints = None
        if self.light_hint is not None:
            assert self.light_hint.shape[1:] == self.image.shape[1:]
            self.light_hints = get_imresize_downsampled(self.light_hint, downsampling_factor=self.downsampling_factor,
                                                        downsampling_number=self.downsampling_number)
            self.light_hints_torch = [np_to_torch(light_hint).type(torch.cuda.FloatTensor) for light_hint in
                                      self.light_hints]
        else:
            self.light_hints = None

def _init_noise(self):
        input_type = 'noise'
        self.left_net_inputs = [get_noise(self.input_depth,
                                          input_type,
                                          (image.shape[2], image.shape[3])).type(torch.cuda.FloatTensor).detach()
                                for image in self.images_torch]
        self.right_net_inputs = self.left_net_inputs
        self.kernel_net_inputs = [get_noise(200, input_type, (1, 1)).type(torch.cuda.FloatTensor)]
        for kernel_net_input in self.kernel_net_inputs:
            kernel_net_input.squeeze_()

def _init_parameters(self):
        self.parameters = [p for p in self.left_net.parameters()] + \
                          [p for p in self.right_net.parameters()] + \
                          [p for p in self.kernel_net.parameters()]
        self.parametersRRD = self.RRD_net.parameters()

def _init_losses(self):
        data_type = torch.cuda.FloatTensor
        self.gngc_loss = YIQGNGCLoss().type(data_type)
        self.exclusion_loss = ExclusionLoss().type(data_type)
        self.l1_loss = nn.L1Loss().type(data_type)
        self.extended_l1_loss = ExtendedL1Loss().type(data_type)
        self.blur_function = StdLoss().type(data_type)
        self.gradient_loss = GradientLoss().type(data_type)
        self.gray_loss = GrayLoss().type(data_type)
        self.grayscale_loss = GrayscaleLoss().type(data_type)
        self.per_loss = PerceptualLoss([0,1,2],[0.5,0.5,0.5],torch.device("cuda" if torch.cuda.is_available() else "cpu")).type(data_type)
        self.color_loss = L_color().type(data_type)
        self.spa_loss = L_spa().type(data_type)
        self.exp_loss = L_exp(16,0.6).type(data_type)
        self.TV_loss = L_TV().type(data_type)

def _init_all(self):
        _init_images(self)
        _init_losses(self)
        _init_nets(self)
        _init_parameters(self)
        _init_noise(self)
