'''ResNet-18 Image classfication for cifar-10 with PyTorch
Using custom convolution layer
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

### Implement 2D convolution layer
class MyConv2d(nn.Module):
    def __init__(self, n_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1):
        super(MyConv2d, self).__init__()

        self.kernel_size = (kernel_size, kernel_size)
        self.kernal_size_number = kernel_size * kernel_size
        self.out_channels = out_channels
        self.dilation = (dilation, dilation)
        self.padding = (padding, padding)
        self.stride = (stride, stride)
        self.n_channels = n_channels
        self.weights = nn.Parameter(torch.Tensor(self.out_channels, self.n_channels, self.kernal_size_number))

    def forward(self, x):
        #print("apply forward method")
        # 依據各項參數得出output feature map的尺寸
        width = self.calculateNewWidth(x)
        height = self.calculateNewHeight(x)
        windows = self.calculateWindows(x)

        result = torch.zeros(
            [x.shape[0] * self.out_channels, width, height], dtype=torch.float32, device='cuda'
        )
        shape = 0
        for channel in range(x.shape[1]):
            for i_convNumber in range(self.out_channels):
                ## matmul for 2-dim and 1-dim
                xx = torch.matmul(windows[channel], self.weights[i_convNumber][channel]).to('cuda')
                xx = xx.view(-1, width, height)  # -1表不確定

                result[i_convNumber * xx.shape[0] : (i_convNumber + 1) * xx.shape[0]] += xx

        result = result.view(x.shape[0], self.out_channels, width, height)
        print("\nResult Shape: ", result.shape)
        return result

    def calculateWindows(self, x):
        windows = F.unfold(
            x, kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation, stride=self.stride
        )

        windows = windows.transpose(1, 2).contiguous().view(-1, x.shape[1], self.kernal_size_number)
        windows = windows.transpose(0, 1)

        return windows

    def calculateNewWidth(self, x):
        return (
            (x.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)
            // self.stride[0]
        ) + 1

    def calculateNewHeight(self, x):
        return (
            (x.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)
            // self.stride[1]
        ) + 1


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()

        self.left = nn.Sequential(
            #nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            MyConv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            #nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            MyConv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchannel)
        )

        '''
        ### my Implement
        self.conv1 = MyConv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1)
        self.left = nn.Sequential(
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(outchannel)
        )
        self.conv2 = MyConv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.BatchNorm2d = nn.BatchNorm2d(outchannel)
        '''
        ###
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                #nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                MyConv2d(inchannel, outchannel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):

        out = self.left(x)

        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()

        self.inchannel = 64
        self.conv1 = nn.Sequential(
            #nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            MyConv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        print("****Finish convolution layer1")
        out = self.layer1(out)
        print("****Finish layer1")
        out = self.layer2(out)
        print("****Finish layer2")
        out = self.layer3(out)
        print("****Finish layer3")
        out = self.layer4(out)
        print("****Finish layer4")
        out = F.avg_pool2d(out, 4)
        print("****Finish avarage pool")
        out = out.view(out.size(0), -1)

        out = self.fc(out)
        print("****Finish FC layer")
        return out


def ResNet18():

    return ResNet(ResidualBlock)
