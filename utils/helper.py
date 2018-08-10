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

def resizedSlice(data_slice, height, width):
    """
    """
    resized_slice = cv.resize(data_slice, (height, width), 
        interpolation = cv.INTER_CUBIC)
    
    return resized_slice
def resizedArray(array, height, width, output_mode):
    """
    """
    depth = array.shape[0]
    if output_mode == 'DHW':
        resized_array = np.zeros((depth, height, width))
        for sl in range(depth):
            data_slice = array[sl,:,:]
            resized_array[sl,:,:] = resizedSlice(data_slice, height, width)
    
    elif output_mode =='DHW1':
        resized_array = np.zeros((depth, height, width, 1))
        for sl in range(depth):
            data_slice = array[sl,:,:]
            resized_array[sl,:,:,0] = resizedSlice(data_slice, height, width)

    return resized_array
