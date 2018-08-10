from config import *
import os
import h5py
import numpy as np
import random
import cv2 as cv


def saveToHdf5(data, dataset_name, save_path, filename):
    """
    Save data into hdf5 file.
    """
    hf = h5py.File(os.path.join(save_path, filename), "w")
    hf.create_dataset(dataset_name, data=data, dtype=np.uint16)
    hf.close()
def loadFromHdf5(load_path, filename, dataset_name):
    """
    Save data into hdf5 file.
    """
    hd = h5py.File(os.path.join(load_path, filename), "r")
    dataset = hd.get(dataset_name)
    data = dataset[...]
    hd.close()
    
    return data

def printShapes():
    """
    Prit dataset shapes
    """
    for i in range(1,101):
        filename = 'patient'+str(i).zfill(3)+'_ED.hdf5'
        x = loadFromHdf5(NP_TRAIN_DIR, filename, 'train')
        print('File: %s shape: %s' % (filename, str(x.shape)))
        filename = 'patient'+str(i).zfill(3)+'_ES.hdf5'
        x = loadFromHdf5(NP_TRAIN_DIR, filename, 'train')
        print('File: %s shape: %s' % (filename, str(x.shape)))

    for i in range(101,151):
        filename = 'patient'+str(i).zfill(3)+'_ED.hdf5'
        x = loadFromHdf5(NP_TEST_DIR, filename, 'test')
        print('File: %s shape: %s' % (filename, str(x.shape)))
        filename = 'patient'+str(i).zfill(3)+'_ES.hdf5'
        x = loadFromHdf5(NP_TEST_DIR, filename, 'test')
        print('File: %s shape: %s' % (filename, str(x.shape)))
