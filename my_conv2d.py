import torch
import torch.nn as nn
import torch.nn.functional as F

class MyConv2d(nn.modules.Conv2d):


    def forward(self, input):

        # Calculate result size
        width = self.calculateNewWidth(input)
        height = self.calculateNewHeight(input)

        # Create Result Tensor
        result = torch.zeros(
            [input.shape[0] * self.out_channels, width, height], dtype=torch.float32, device='cuda'
        )
        #print("result size: ", result.shape)
        # Padding
        input = F.pad(input, (self.padding[0], self.padding[0], self.padding[1], self.padding[1]))
        #print(input.shape)
        #print(result.shape)

        for n1 in range(input.shape[0]):
            for n2 in range(self.out_channels):
                for rr in range(result.shape[1]):
                    for rc in range(result.shape[2]):
                        r = rr*self.stride[1] ; c = rc*self.stride[0]
                        result[n1*input.shape[0]+n2][rr][rc] = 0
                        for ch in range(self.in_channels):
                            result[n1*input.shape[0]+n2][rr][rc] += \
                            input[n1][ch][r][c]*self.weight[n2][ch][0][0] + input[n1][ch][r][c+1]*self.weight[n2][ch][0][1] + input[n1][ch][r][c+2]*self.weight[n2][ch][0][2] + \
                            input[n1][ch][r+1][c]*self.weight[n2][ch][1][0] + input[n1][ch][r+1][c+1]*self.weight[n2][ch][1][1] + input[n1][ch][r+1][c+2]*self.weight[n2][ch][1][2] + \
                            input[n1][ch][r+2][c]*self.weight[n2][ch][2][0] + input[n1][ch][r+2][c+1]*self.weight[n2][ch][2][1] + input[n1][ch][r+2][c+2]*self.weight[n2][ch][2][2]

        result = result.view(input.shape[0], self.out_channels, width, height)
        return result
    def getWindows(self, input):
        pass

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
