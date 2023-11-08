import torch.utils.data
from torch.nn import functional as F

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple


class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)


class Generator(nn.Module):
    '''
    Generator Class
    Values:
        input_dim: the dimension of the input vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, args):
        super(Generator, self).__init__()
        self.init_size = 1
        self.n_classes = args.n_classes
        self.ndim = args.z_dim
        self.n_info = args.n_info
        self.im_chan = args.channels

        self.input_dim = self.ndim + self.n_info + self.n_classes
        
        self.embedding_c = nn.Embedding(num_embeddings=self.n_classes, embedding_dim=self.n_classes)
        nn.init.eye_(self.embedding_c.weight)
        self.embedding_c.weight.requires_grad_(False)
        # Build the neural network
        self.l1 = nn.Sequential(nn.Linear(self.input_dim, 768))
        
        self.conv_blocks = nn.Sequential(
            self.make_gen_block(768, 384),
            self.make_gen_block(384, 256),
            self.make_gen_block(256, 192),
            self.make_gen_block(192, 64),
            self.make_gen_block(64, self.im_chan, kernel_size=8, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=5, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
            )                  

        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, bias=False),
                nn.Tanh(),
            )

    def forward(self, noise, labels, info):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, input_dim)
        '''

        c = self.embedding_c(labels.long()) # N --> N x C
        # i = self.embedding_c(info) # N --> N x C
        input = torch.cat((noise, c, info), dim=1)

        x = self.l1(input)
        x = x.reshape(x.shape[0], 768, self.init_size, self.init_size)
        x = self.conv_blocks(x)
        return x
    
    
class Discriminator(nn.Module):
    '''
    Critic Class
    Values:
      im_chan: the number of channels in the images, fitted for the dataset used, a scalar
            (MNIST is black-and-white, so 1 channel is your default)
      hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, args):
        super().__init__()
        self.n_classes = args.n_classes
        self.ndim = args.z_dim
        self.im_chan = args.channels
        self.input_dim = self.im_chan
        self.n_info = args.n_info

        # one-hot embedding_c
        # self.embedding_c = nn.Embedding(num_embeddings=self.n_classes, embedding_dim=self.n_classes)
        # nn.init.eye_(self.embedding_c.weight)
        # self.embedding_c.weight.requires_grad_(False)

        self.disc = nn.Sequential(
            self.make_disc_block(self.input_dim, 16, padding_mode="same"),
            self.make_disc_block(16, 32, stride=1),
            self.make_disc_block(32, 64, padding_mode="same"),
            self.make_disc_block(64, 128, stride=1),
            self.make_disc_block(128, 256, padding_mode="same"),
            self.make_disc_block(256, 512, stride=1),
        )

        self.fs = nn.Sequential(
            nn.Linear(86528, 1),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(86528, self.n_classes),
            nn.Softmax(dim=1)
        )
        
        self.fi = nn.Sequential(
            nn.Linear(86528, self.n_info),
        )


    def make_disc_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False, padding_mode="valid"):
        '''
        Function to return a sequence of operations corresponding to a discriminator block of the DCGAN; 
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if padding_mode == "same":
            return nn.Sequential(
                Conv2d(input_channels, output_channels, kernel_size, stride, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.5)
            )

        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        '''

        input = image
        x = self.disc(input)
        # print(x.shape)
        x = x.reshape(x.shape[0], -1)
        
        s = self.fs(x)
        c = self.fc(x)
        i = self.fi(x)

        return s.view(len(x), -1), c, i