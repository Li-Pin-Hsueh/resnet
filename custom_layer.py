import torch
from torch import nn
from torch.nn import functional as F

class MyConv(nn.Conv2d):


    def forward(self, inputs):
        wout = self.calculateNewWidth(inputs)
        hout = self.calculateNewHeight(inputs)
        ofm_size = wout

        result = torch.zeros(
            [inputs.shape[0] * self.out_channels, wout, hout], dtype=torch.float32, device='cuda'
        )

        for n1 in range(inputs.shape[0]):
            for n2 in range(self.out_channels):
                row = 0 ; col = 0
                result[n1*self.out_channels+n2][row][col] = 0
                for channel in range(inputs.shape[1]):
                    ## Get window
                    r = 0 ; c = 0
                    while( r < inputs.shape[2] ):
                        while( c < inputs.shape[3] ):
                            if( r+self.kernel_size[0] <= inputs.shape[2] and c+self.kernel_size[1] <= inputs.shape[3]):
                                window = torch.ones([3,3], dtype = torch.float32, device='cuda')
                                #window = inputs[n1][n2][r:r+self.kernel_size[0] ][c:c+self.kernel_size[1] ]
                                result[n1*self.out_channels+n2][row][col] += torch.mat(window, self.weight[n2][c].to('cuda'))
                            r += 1 ; c += 1
                    row += 1 ; col += 1



        return result

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
