import torch
import numpy as np
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from scipy.signal import correlate2d, convolve2d

'''
Implement custom module
'''
class ScipyConv2d(Module):
    def __init__(self, filter_width, filter_height):
        super(ScipyConv2d, self).__init__()
        # 兩個可學習的 Parameters，我們將會對這兩個參數求梯度
        self.filter = Parameter(torch.randn(filter_width, filter_height))
        self.bias = Parameter(torch.randn(1, 1))

    def forward(self, input):
        # 呼叫 ScipyConv2dFunction 的 apply
        return ScipyConv2dFunction.apply(input, self.filter, self.bias)

module = ScipyConv2d(2, 2)
print("Filter and bias: ", list(module.parameters()))

'''
Implement custom autograd function
'''
class ScipyConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, filter, bias):
        # detach so we can cast to NumPy
        input, filter, bias = input.detach(), filter.detach(), bias.detach()
        # scipy correlate2d method
        result = correlate2d(input.numpy(), filter.numpy(), mode='valid')
        result += bias.numpy()
        ctx.save_for_backward(input, filter, bias)
        return torch.as_tensor(result, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach()
        input, filter, bias = ctx.saved_tensors
        grad_output = grad_output.numpy()
        grad_bias = np.sum(grad_output, keepdims=True)
        grad_input = convolve2d(grad_output, filter.numpy(), mode='full')
        grad_filter = correlate2d(input.numpy(), grad_output, mode='valid')
        return torch.from_numpy(grad_input), \
               torch.from_numpy(grad_filter).to(torch.float), \
               torch.from_numpy(grad_bias).to(torch.float)



input = torch.randn(5, 5, requires_grad=True)
output = module(input)
output.backward(torch.randn(4, 4))
print("Gradient for the input map: ", input.grad)
print("Output from the convolution: ", output)
