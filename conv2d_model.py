import torch
import torch.nn as nn
import torch.nn.functional as F

import time
class MyConv2d(nn.modules.Conv2d):


    def forward(self, input):
        # Specific weights
        self.weight = nn.Parameter(torch.ones_like(self.weight))



        # Calculate result size
        width = self.calculateNewWidth(input)
        height = self.calculateNewHeight(input)

        # Create Result Tensor
        result = torch.zeros(
            [input.shape[0] , self.out_channels, width, height], dtype=torch.float32#, device='cpu'
        )
        #print("result size: ", result.shape)
        # Padding
        input = F.pad(input, (self.padding[0], self.padding[0], self.padding[1], self.padding[1]), value=1)#.to('cuda')

        #print(input.shape)
        #print(result.shape)
        print("\n2D Convolution Layer....")
        #print("input size: ", input.shape)
        #print("weight size: ", self.weight.shape)
        #print("out channels: ", self.out_channels)
        #print("result sieze: ", result.shape)
        #print("stride: ", self.stride)
        print("Input Size: ", input.shape)

        N = input.shape[0]
        CH = input.shape[1]
        H = input.shape[2]
        W = input.shape[3]

        C = self.weight.shape[0]
        FR = self.weight.shape[2]
        FC = self.weight.shape[3]

        for n1 in range(N):
            for n2 in range(self.out_channels):
                for rr in range(result.shape[2]):
                    for rc in range(result.shape[3]):

                        r = rr*self.stride[1] ; c = rc*self.stride[0]

                        for ch in range(CH):
                            result[n1][n2][rr][rc] += \
                            input[n1][ch][r][c]*self.weight[n2][ch][0][0] + input[n1][ch][r][c+1]*self.weight[n2][ch][0][1] + input[n1][ch][r][c+2]*self.weight[n2][ch][0][2] + \
                            input[n1][ch][r+1][c]*self.weight[n2][ch][1][0] + input[n1][ch][r+1][c+1]*self.weight[n2][ch][1][1] + input[n1][ch][r+1][c+2]*self.weight[n2][ch][1][2] + \
                            input[n1][ch][r+2][c]*self.weight[n2][ch][2][0] + input[n1][ch][r+2][c+1]*self.weight[n2][ch][2][1] + input[n1][ch][r+2][c+2]*self.weight[n2][ch][2][2]


        '''
        r1 = -1 ; r2 = -1
        for n in range(N):
            for row in range(0, H+1, self.stride[0]):
                r1 += 1 ; r2 = -1
                for col in range(0, W+1, self.stride[1]):
                    r2 += 1
                    for ch in range(CH):
                        for i in range(C):
                            for fr in range(FR):
                                for fc in range(FC):
                                    if( row+fr < H and col+fc < W and r1 < result.shape[2] and r2 < result.shape[3]):
                                        result[n][i][r1][r2] +=     \
                                        self.weight[i][ch][fr][fc] * \
                                        input[n][ch][row+fr][col+fc]
        '''


        #print("Finished Conv2d Computation")

        #result = result.view(input.shape[0], self.out_channels, width, height)
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



class MyNet(nn.Module):

    def __init__(self, inchannel, outchannel, stride=1):
        super(MyNet, self).__init__()
        '''
        self.left = nn.Sequential(
            #nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            MyConv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
            MyConv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
            MyConv2d(outchannel, 64, kernel_size=3, stride=1, padding=1),
        )
        '''
        self.conv1 = MyConv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1)
        self.conv2 = MyConv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.conv3 = MyConv2d(outchannel, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        t1 = time.time()
        x = self.conv1(x)
        t2 = time.time()
        print('Convolution Layer 1 Time Consumed: ' + str(round(t2-t1, 2)) + ' seconds')
        print("*********************")
        x = self.conv2(x)
        t3 = time.time()
        print('Convolution Layer 2 Time Consumed: ' + str(round(t3-t2, 2)) + ' seconds')
        print("*********************")
        out = self.conv3(x)
        t4 = time.time()
        print('Convolution Layer 3 Time Consumed: ' + str(round(t4-t3, 2)) + ' seconds')
        print("*********************")

        #out = F.relu(out)
        return out

#net = MyConv2d(3, 3, 3, padding=1)
net = MyNet(3, 4)#.to('cuda')
x = torch.ones(1, 3, 32, 32)#.to('cuda')


t5 = time.time()
out = net(x)
t6 = time.time()
print("\nInput Size: ", x.shape)
print("\nOutput Size: ", out.shape)
print('\nTotal Time Consumed: ' + str(round(t6-t5, 2)) + ' seconds')
print(out[0][0])
