import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from scipy.signal import correlate2d, convolve2d

'''
Implement custom module
'''
class MyConv2d(Module):
    def __init__(self, n_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1):
        super(MyConv2d, self).__init__()
        self.kernel_size = (kernel_size, kernel_size)
        self.out_channels = out_channels
        self.dilation = (dilation, dilation)
        self.padding = (padding, padding)
        self.stride = (stride, stride)
        self.n_channels = n_channels
        self.weight = nn.Parameter(torch.randn(self.out_channels, self.n_channels, self.kernel_size[0], self.kernel_size[1]))


    def forward(self, input):
        # 呼叫 ScipyConv2dFunction 的 apply
        return F.conv2d(input, self.weight, padding=self.padding, stride=self.stride)


'''
Implement custom autograd function

class MyConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input):
        pass
    @staticmethod
    def backward(ctx, grad_output):
        pass
'''

module = MyConv2d(3, 1, 3)
print("Filter and bias: ", list(module.parameters()))

input = torch.randn(3, 3, 5, 5, requires_grad=True)
output = module(input)
output.backward(torch.randn(3,1,3,3))
print("Gradient for the input map: ", input.grad)
print("Output from the convolution: ", output)
