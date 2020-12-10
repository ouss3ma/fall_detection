import os
import sys
import pickle
import linecache
import argparse
import numpy as np
from numpy.lib.format import open_memmap
import torch
import torch.nn as nn
import torch.nn.functional as F
from pandas import read_csv

cuda_available = torch.cuda.is_available()
print(cuda_available)

subseq=30


VALID_RATIO = 0.10
TRAIN_RATIO = 1.0 - VALID_RATIO
LEARNING_RATE = 0.00001
KERNEL_SIZE = 5
PAD = 1
BATCH_SIZE = 128
MAX_EPOCH = 186

SAVE_MODEL = False  # Save model after each epoch
LOAD_MODEL = False # Skip training phase
SAVE_PATH = 'BestModel.pwf'

TRAIN_DIR =''
TEST_DIR = ''


#load data
def read_skeleton(file,subseq):
    f = open(file, 'r')
    Lines = f.readlines()
    t = 1
    data = []
    for line in Lines:
        t += 1
        line = line.split()[0:75] #75=25*3

        data += line

    print(np.array(data).shape)
    data = data[0:346500]
    data = np.array(data)
    print(data.shape)
    samples=154
    data = data.reshape(samples,subseq,75)

    return data












if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PKU-MMD Data Converter.')
    parser.add_argument('--data_path',
                        default='/home/oussema/code/echantillon')

    parser.add_argument('--out_folder', default='/home/oussema/code/st-gcn/data/PKU')

    arg = parser.parse_args()

    #net = Classifier()
    file='/home/oussema/code/PKU-MMD/PKU_Skeleton_Renew/0002-L.txt'
    read_skeleton(file,subseq)