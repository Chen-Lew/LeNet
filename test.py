from model import LeNet
import torch
from PIL import Image
import numpy
import torchvision.transforms as transforms

def main():
    transform = transforms.Compose([
        transforms.Resize((32,32)),#易错点：接收的是一个tuple
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 加载权重weight已经训练好的模型
    net = LeNet()
    net.load_state_dict(torch.load('LeNet.pth'))


    im = Image.open('3.jpeg')
    im = transform(im)
    im = torch.unsqueeze(im,dim=0)

    with torch.no_grad():
        outputs = net(im)
        pred = torch.max(outputs,dim=1)[1].numpy()
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(classes[int(pred)])





if __name__ == '__main__':
    main()



