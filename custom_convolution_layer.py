import torch
from torch import nn
from torch.nn import functional as F

class MyConv(nn.Conv2d):

    def __init__(self, n_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1):
        super(MyConv, self).__init__(n_channels, out_channels, kernel_size)

        self.kernel_size = (kernel_size, kernel_size)
        self.kernel_size_number = kernel_size * kernel_size
        self.out_channels = out_channels
        self.dilation = (dilation, dilation)
        self.padding = (padding, padding)
        self.stride = (stride, stride)
        self.n_channels = n_channels
        self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.n_channels, self.kernel_size_number))

    def forward(self, input):
        wout = self.calculateNewWidth(input)
        hout = self.calculateNewHeight(input)
        windows = self.calculateWindows(input)

        result = torch.zeros(
            [input.shape[0] * self.out_channels, wout, hout], dtype=torch.float32, device='cuda'
        )

        for channel in range(input.shape[1]):
            for i_convNumber in range(self.out_channels):
                ## matmul for 2-dim and 1-dim
                xx = torch.matmul(windows[channel], self.weight[i_convNumber][channel]).to('cuda')
                xx = xx.view(-1, wout, hout)  # -1表不確定
                result[i_convNumber * xx.shape[0] : (i_convNumber + 1) * xx.shape[0]] += xx

        result = result.view(input.shape[0], self.out_channels, wout, hout)
        return result


    def calculateWindows(self, input):

        windows = F.unfold(
            input, kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation, stride=self.stride
        )

        windows = windows.transpose(1, 2).contiguous().view(-1, input.shape[1], self.kernel_size_number)
        windows = windows.transpose(0, 1)

        return windows

    def calculateNewWidth(self, input):
        return (
            (input.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)
            // self.stride[0]
        ) + 1

    def calculateNewHeight(self, input):
        return (
            (input.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)
            // self.stride[1]
        ) + 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('*******\nUsing device:', torch.cuda.get_device_name(device))

net = nn.Sequential(
    MyConv(3, 1, kernel_size=3, stride=1, padding=1, dilation=1),
)

x = torch.randn(1, 3, 64, 64)
out = net(x)
out.mean().backward()
print("output shape", out.shape)
