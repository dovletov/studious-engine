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
def printMeanShape():
    """
    Print mean z,x,y sizes for training dataset.
    """
    mx = 0
    my = 0
    mz = 0
    for i in range(1,101):
        filename = 'patient'+str(i).zfill(3)+'_ED.hdf5'
        x = loadFromHdf5(NP_TRAIN_DIR, filename, 'train')
        mz += x.shape[0]
        mx += x.shape[1]
        my += x.shape[2]
        filename = 'patient'+str(i).zfill(3)+'_ES.hdf5'
        x = loadFromHdf5(NP_TRAIN_DIR, filename, 'train')
        mz += x.shape[0]
        mx += x.shape[1]
        my += x.shape[2]
    mz = int(mz/200)
    mx = int(mx/200)
    my = int(my/200)
    print("Mean z = %z" % mz)
    print("Mean x = %z" % mx)
    print("Mean y = %z" % my)

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
def getRandSlices(subset_name):
    """
    Returns randoms slice for image and corresponding GT image.
    """
    mode = random.randint(0, 1)
    if subset_name == 'tr':
        pid = random.randint(1,80)
    elif subset_name == 'vl':
        pid = random.randint(81,100)

    if mode == 0:
        x_name = 'patient' + str(pid).zfill(3)+'_ED.hdf5'
        y_name = 'patient' + str(pid).zfill(3)+'_ED_gt.hdf5'
    else:
        x_name = 'patient' + str(pid).zfill(3)+'_ES.hdf5'
        y_name = 'patient' + str(pid).zfill(3)+'_ES_gt.hdf5'

    x = loadFromHdf5(NP_TRAIN_DIR, x_name, 'train')
    y_ = loadFromHdf5(NP_TRAIN_DIR, y_name, 'train_gt')

    depth = x.shape[0]
    sid = random.randint(0,depth-1)

    return x[sid,:,:], y_[sid,:,:]
def resizedSlice(data_slice, height, width):
    """
    """
    resized_slice = cv.resize(data_slice, (height, width), 
        interpolation = cv.INTER_CUBIC)
    
    return resized_slice
def formRandBatch(batch_size, height, width, subset_name):
    """
    Forms BHW1 image batch and corresponding BHW GT batch.
    """
    x_batch, y_batch = getRandSlices(subset_name=subset_name)
    if batch_size > 1:
        x_batch = resizedSlice(x_batch, height, width)
        y_batch = resizedSlice(y_batch, height, width)
        
        # add B axe
        x_batch = np.expand_dims(x_batch, 0)
        y_batch = np.expand_dims(y_batch, 0)
    
        for i in range(batch_size-1):
            x, y = getRandSlices(subset_name=subset_name)
            x = resizedSlice(x, height, width)
            y = resizedSlice(y, height, width)
            x = np.expand_dims(x,0)
            y = np.expand_dims(y,0)
            x_batch = np.concatenate((x_batch, x), axis=0)
            y_batch = np.concatenate((y_batch, y), axis=0)

        x_batch = np.expand_dims(x_batch, 3)

    return x_batch, y_batch
