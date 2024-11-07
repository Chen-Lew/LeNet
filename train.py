from model import LeNet

import torch
import torch.nn as nn
import torch.functional as F

import torchvision
import torchvision.transforms as transforms

import torch.optim as optim

def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./dataset',
                                                 train=True,
                                                 transform=transform,
                                                 download=True)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                  batch_size = 36,
                                                  shuffle = True,
                                                  num_workers=0)
    val_dataset = torchvision.datasets.CIFAR10(root='./dataset',
                                               train=False,
                                               transform=transform,
                                               download=True)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size = 5000,
                                                shuffle=True,
                                                num_workers = 0)

    val_data_iter = iter(val_dataloader)
    val_image , val_label = next(val_data_iter)

    net = LeNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.01)

    for epoch in range(10):
        running_loss = 0.0
        for step , data in enumerate(train_dataloader):
            inputs , labels = data
            outputs = net(inputs)
            optimizer.zero_grad()
            loss = loss_function(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 500 == 499:
                with torch.no_grad():
                    outputs = net(val_image) #[batch_size , category_num10]
                    pred = torch.max(outputs,dim=1)[1]
                    accuracy = torch.eq(pred,val_label).sum().item() / val_label.size(0)

                    print('epoch=%d  step=%d  loss_mean=%f   accuracy=%f' %
                          (epoch+1,step+1,running_loss/500,accuracy))
                    running_loss = 0.0
    print('finished')

    save_pth = './LeNet.pth'
    torch.save(net.state_dict(),save_pth)






if __name__ == '__main__':
    main()