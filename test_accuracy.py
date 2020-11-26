import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from resnet import ResNet18
import torchvision
import torchvision.transforms as transforms

from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
signal(SIGPIPE, SIG_IGN)


#for item in new:
#    print(item)



transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('*******\nUsing device:', torch.cuda.get_device_name(device))
torch.cuda.empty_cache()

model = ResNet18().to(device)
model.load_state_dict(pretrained_weights)


correct = 0 ; total = 0 ; count = 0

for data in testloader:
    count += 1
    model.eval()
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    # 取得分最高的那个类 (outputs.data的索引号)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    print('[' + str(count) + ']' + 'Test classfication accuracy:%.2f' % (100 * correct / total))
