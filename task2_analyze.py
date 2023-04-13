#Syed Azhar Hussain Quadri
# CS 5330 Project 5

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import trunc
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as T
import torch
from torch.autograd import Variable
import torch.nn as tn
import torch.nn.functional as tnf
import torch.optim as t0
from task1_train import Net


class Net(tn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = tn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = tn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = tn.Dropout2d()
        self.fc1 = tn.Linear(320, 50)
        self.fc2 = tn.Linear(50, 10)

    def forward(self, x):
        x = tnf.relu(tnf.max_pool2d(self.conv1(x), 2))
        x = tnf.relu(tnf.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = tnf.relu(self.fc1(x))
        x = tnf.dropout(x, training=self.training)
        x = self.fc2(x)
        return tnf.log_softmax(x)

class Submodel(Net):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override the forward method
    def forward( self, x ):
        x = tnf.relu( tnf.max_pool2d( self.conv1(x), 2 ) ) # relu on max pooled results of conv1
        x = tnf.relu( tnf.max_pool2d( self.conv2_drop( self.conv2(x)), 2 ) ) # relu on max pooled results of dropout of conv2
        return x


def visualize(weight):
    # print(weight.shape)
    # print(weight[5])
    weight = weight - weight.min()
    weight = weight / weight.max()
    plt.figure(figsize=(5,6))
    for i, filter in enumerate(weight):
        plt.subplot(5, 5, i+1) 
        plt.imshow(filter[0, :, :].detach(), cmap='viridis')
        plt.axis('off')
        plt.title("Filter {}".format(i))
        plt.xticks=[]
        plt.yticks=[]
        
    plt.show()

def visualizeFilters(weights, img):
    with torch.no_grad():
        var = img.clone()
        var.unsqueeze_(0)
        var = Variable(var,requires_grad = False)
        src_image = var[0][0].cpu().detach().numpy()
        
        print(src_image.shape)
        imgplot = plt.imshow(src_image[0])
        plt.show()

       
        f,axarr = plt.subplots(10,2)
        
        axarr[0,0].imshow(weights.cpu().detach().numpy()[0][0], cmap='gray')
        axarr[0,1].imshow(cv2.filter2D(src_image[0], -1, weights.cpu().detach().numpy()[0][0]), cmap='gray')

        axarr[1,0].imshow(weights.cpu().detach().numpy()[1][0], cmap='gray')
        axarr[1,1].imshow(cv2.filter2D(src_image[0], -1, weights.cpu().detach().numpy()[1][0]), cmap='gray')
        
        axarr[2,0].imshow(weights.cpu().detach().numpy()[2][0], cmap='gray')
        axarr[2,1].imshow(cv2.filter2D(src_image[0], -1, weights.cpu().detach().numpy()[2][0]), cmap='gray')

        axarr[3,0].imshow(weights.cpu().detach().numpy()[3][0], cmap='gray')
        axarr[3,1].imshow(cv2.filter2D(src_image[0], -1, weights.cpu().detach().numpy()[3][0]), cmap='gray')

        axarr[4,0].imshow(weights.cpu().detach().numpy()[4][0], cmap='gray')
        axarr[4,1].imshow(cv2.filter2D(src_image[0], -1, weights.cpu().detach().numpy()[4][0]), cmap='gray')

        axarr[5,0].imshow(weights.cpu().detach().numpy()[5][0], cmap='gray')
        axarr[5,1].imshow(cv2.filter2D(src_image[0], -1, weights.cpu().detach().numpy()[5][0]), cmap='gray')

        axarr[6,0].imshow(weights.cpu().detach().numpy()[6][0], cmap='gray')
        axarr[6,1].imshow(cv2.filter2D(src_image[0], -1, weights.cpu().detach().numpy()[6][0]), cmap='gray')

        axarr[7,0].imshow(weights.cpu().detach().numpy()[7][0], cmap='gray')
        axarr[7,1].imshow(cv2.filter2D(src_image[0], -1, weights.cpu().detach().numpy()[7][0]), cmap='gray')

        axarr[8,0].imshow(weights.cpu().detach().numpy()[8][0], cmap='gray')
        axarr[8,1].imshow(cv2.filter2D(src_image[0], -1, weights.cpu().detach().numpy()[8][0]), cmap='gray')

        axarr[9,0].imshow(weights.cpu().detach().numpy()[9][0], cmap='gray')
        axarr[9,1].imshow(cv2.filter2D(src_image[0], -1, weights.cpu().detach().numpy()[9][0]), cmap='gray')
    plt.show()


	

def main():
    network = Net()
    network.load_state_dict(torch.load("mnistModel.h5"))
    network.eval()
    weights=network.conv1.weight
    visualize(weights)

    weights = weights - weights.min()
    weights = weights / weights.max()

    # filter=weights[0,:,:].cpu().detach().numpy()

    #get first training example image
    train_images = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files', train=True, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                  (0.1307,), (0.3081,))
                              ])),
    batch_size=64, shuffle=True)
    imgs = enumerate(train_images)
    i,(img_data,j) = next(imgs)
    
    visualizeFilters(weights,img_data)

    #Part C
    sub = Submodel()
    sub_opt = t0.SGD(sub.parameters(), lr=0.01, momentum=0.5)
    sub.load_state_dict(torch.load("mnistModel.h5"))
    sub_opt.load_state_dict(torch.load("./optimizer.pth"))
    sub.eval()

    sub_weights=sub.conv1.weight
    sub_weights = sub_weights - sub_weights.min()
    sub_weights = sub_weights / sub_weights.max()

    with torch.no_grad():
      sub_opt = sub(img_data)
    
    m = tn.Dropout(p=0.2)
    input = torch.randn(20,16)
    output = m(input)
    print(output.size())

    visualizeFilters(sub_weights,img_data)
    
    # truncated(sub,img_data)
if __name__ == "__main__":
    main()
  

