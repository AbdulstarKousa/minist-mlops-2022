import numpy as np
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

def mnist():
    ''' Function to load MINST data. For each batch:
        > images.shape: [64, 28, 28] , 
        > labels.shape: [64]
    '''

    # data path
    path = os.path.join('..', 'data', 'corruptmnist')

    # test data
    test_imgs = np.load(os.path.join(path, 'test.npz'))['images']
    test_labs = np.load(os.path.join(path, 'test.npz'))['labels']
    test_imgs  = torch.Tensor(test_imgs)
    test_labs  = torch.Tensor(test_labs).type(torch.LongTensor)

    # train data
    filenames       = ['train_0.npz', 'train_1.npz', 'train_2.npz', 'train_3.npz', 'train_4.npz']
    train_imgs_lst   = []
    train_labs_lst = []
    for filename in filenames:
        train_imgs_lst.append(np.load(os.path.join(path, filename))['images'])
        train_labs_lst.append(np.load(os.path.join(path, filename))['labels'])
    train_imgs = np.concatenate(tuple(train_imgs_lst))
    train_labs = np.concatenate(tuple(train_labs_lst))
    train_imgs = torch.Tensor(train_imgs)
    train_labs = torch.Tensor(train_labs).type(torch.LongTensor)

    # to dataset
    train_ds = TensorDataset(train_imgs, train_labs)
    test_ds  = TensorDataset(test_imgs, test_labs)

    # to dataloader
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl  = DataLoader(test_ds , batch_size=64, shuffle=False)   

    # # plot
    # import matplotlib.pyplot as plt
    # dataiter = iter(train_dl)
    # images, labels = dataiter.next()
    # print(images.shape)
    # print(labels.shape)
    # plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');
    # plt.show()
    # return 
    return train_dl, test_dl



if __name__ == '__main__':
    mnist()