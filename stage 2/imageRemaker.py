'''
    putting images through nn's to get idea about what the architectures do to the images
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
from torch.autograd import Variable

import os
import time
import random
import numpy as np
from tqdm import tqdm
from matplotlib import image
from matplotlib import pyplot as plt
import logging

# logger = logging.getLogger()
# old_level = logger.level
# logger.setLevel(100)

import architectures as arcs

def train(net, criterion, optimizer, im, epochs, watch=True, watchInterval=10):
    net.train()

    imFlat = im.flatten()
    imLen = imFlat.shape[0]
    print(imLen)
    print(im.shape)

    if watch:
        fig = plt.figure()
        data = np.zeros(im.shape)
        implt = plt.imshow(data)
        plt.show(block=False)
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    imFlat = imFlat.to(device)

    pbar = tqdm(range(epochs))
    for i in pbar:
        # barDescription = "Game %s, Records: Reward %s, Snake Length %s, Steps %s, New Visits %s" % (i, str(topReward)[:4], longestSnake, longestGame, mostNewVisited)
        # pbar.set_description(barDescription)
        optimizer.zero_grad()
        output = net(imFlat)

        if watch:
            if i % watchInterval == 0:
                outputImage = output.view(im.shape).detach().to("cpu").clamp(0.0, 1.0)
                implt.set_data(outputImage)
                fig.canvas.draw()
                time.sleep(0.01)

        loss = criterion(output, imFlat)
        loss.backward(retain_graph=True)
        optimizer.step()
    return False

def main():
    #   load image
    fileName = 'im_20percent'
    ext = '.png'
    imRaw = image.imread(fileName + ext)
    print(imRaw.dtype)
    print(imRaw.shape)
    # pyplot.imshow(im)
    # pyplot.show()

    im = torch.tensor(imRaw, dtype=torch.float32)
    imFlat = im.flatten()
    imLen = imFlat.shape[0]
    print(imLen)
    print(im.shape)

    #   make net
    net = arcs.TinyNet(imLen, 8, imLen)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    #   cuda shit
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    net.to(device)
    im.to(device)

    #   train
    train(net=net, criterion=criterion, optimizer=optimizer, im=im, epochs=1000, watch=True, watchInterval=100)

    #   evaluate results
    #   #   bring back to cpu
    cpuDevice = torch.device("cpu")
    imFlat.to(cpuDevice)
    net.to(cpuDevice)

    #   run through nn
    net.eval()
    output = net(imFlat)
    outputImage = output.view(im.shape).detach().numpy()
    outputImage = outputImage.clip(0.0, 1.0)
    displayImage = outputImage.copy()
    print(outputImage.dtype)
    print(outputImage.shape)

    outputImage = np.rollaxis(outputImage, 2, 0)
    newOutputImage = outputImage.copy()
    idx = [2, 0, 1]
    newOutputImage = newOutputImage[idx]
    print(outputImage.dtype)
    print(outputImage.shape)
    # outputImage = outputImage.transpose(3, 0, 1, 2)
    newOutputImage = np.rollaxis(newOutputImage, 0, 3)
    print(newOutputImage.dtype)
    print(newOutputImage.shape)

    #   save
    image.imsave(fileName + "_nn_out" + ext, displayImage)
    
    #   show
    plt.imshow(displayImage)
    plt.show()

if __name__ == '__main__':
    main()