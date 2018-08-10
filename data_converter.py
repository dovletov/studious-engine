from config import *
from utils.helper import *

import SimpleITK as sitk
import glob

"""
Convert original '*.nii.gz' data files into numpy arrays and save them as hdf5.
For each of training case there are four files (two images and two GT images):
E.g. for patient001:
    patient001_frame01.nii.gz
    patient001_frame01_gt.nii.gz
    patient001_frame12.nii.gz
    patient001_frame12_gt.nii.gz
Files with lower 'frame_id' are saved with '_ED' ending.
Files with higher 'frame_id' are saved with '_ES' ending.
"""

tr_input_path = TRAIN_DIR
tr_output_path = NP_TRAIN_DIR
tr_range = (1,100)

ev_input_path = TEST_DIR
ev_output_path = NP_TEST_DIR
ev_range = (101,150)

dataset_names = ['train', 'train_gt', 'test']

os.mkdir(tr_output_path)
os.mkdir(ev_output_path)

# Convert training dataset
for p_id in range(*tr_range):
    folder_name = 'patient' + str(p_id).zfill(3)
    folder_path = os.path.join(tr_input_path, folder_name)

    pattern = os.path.join(folder_path, folder_name) + '_frame*'
    files = glob.glob(pattern)
    
    print(pattern)
    if len(files) == 4:
        pass
        # for i in range(len(files)):
        #     print("\t"+files[i])
    else:
        raise ValueError("Wrong number of input files.")
        exit(1)

    ed_found = False
    idx = 0
    while not ed_found:
        filename_ed_X = os.path.join(folder_path, folder_name + \
            '_frame'+str(idx).zfill(2)+'.nii.gz')
        filename_ed_y = os.path.join(folder_path, folder_name + \
            '_frame'+str(idx).zfill(2)+'_gt.nii.gz')

        if os.path.isfile(filename_ed_X) and os.path.isfile(filename_ed_y):
            ed_found=True
            print("ED files are found")
            print('\t' + filename_ed_X)
            print('\t' + filename_ed_y)
        idx += 1

    es_found = False
    while not es_found:
        filename_es_X = os.path.join(folder_path, folder_name + \
            '_frame'+str(idx).zfill(2)+'.nii.gz')
        filename_es_y = os.path.join(folder_path, folder_name + \
            '_frame'+str(idx).zfill(2)+'_gt.nii.gz')

        if os.path.isfile(filename_es_X) and os.path.isfile(filename_es_y):
            es_found=True
            print("ES files are found")
            print('\t' + filename_es_X)
            print('\t' + filename_es_y)
        idx += 1

    
    reader = sitk.ImageFileReader()
    
    reader.SetFileName(filename_ed_X)
    image = reader.Execute()
    nda = sitk.GetArrayFromImage(image)
    # print(nda.shape)
    # for k in reader.GetMetaDataKeys():
    #   v = reader.GetMetaData(k)
    #   print("({0}) = = \"{1}\"".format(k,v))
    filename = folder_name + '_ED.hdf5'
    saveToHdf5(nda, dataset_names[0], tr_output_path, filename)


    reader.SetFileName(filename_ed_y)
    image = reader.Execute()
    nda = sitk.GetArrayFromImage(image)
    # print(nda.shape)
    # print(np.unique(nda, return_counts=True))
    if (nda.max()>4):
        raise ValueError("Something wrong with gt image")
        exit(1)
    filename = folder_name + '_ED_gt.hdf5'
    saveToHdf5(nda, dataset_names[1], tr_output_path, filename)


    reader.SetFileName(filename_es_X)
    image = reader.Execute()
    nda = sitk.GetArrayFromImage(image)
    # print(nda.shape)
    # for k in reader.GetMetaDataKeys():
    #   v = reader.GetMetaData(k)
    #   print("({0}) = = \"{1}\"".format(k,v))
    filename = folder_name + '_ES.hdf5'
    saveToHdf5(nda, dataset_names[0], tr_output_path, filename)


    reader.SetFileName(filename_es_y)
    image = reader.Execute()
    nda = sitk.GetArrayFromImage(image)
    # print(nda.shape)
    # print(np.unique(nda, return_counts=True))
    if (nda.max()>4):
        raise ValueError("Something wrong with gt image")
        exit(1)
    filename = folder_name + '_ES_gt.hdf5'
    saveToHdf5(nda, dataset_names[1], tr_output_path, filename)

# Convert testing dataset
for p_id in range(*ev_range):
    folder_name = 'patient' + str(p_id).zfill(3)
    folder_path = os.path.join(ev_input_path, folder_name)

    pattern = os.path.join(folder_path, folder_name) + '_frame*'
    files = glob.glob(pattern)
    
    print(pattern)
    if len(files) == 2:
        pass
        # for i in range(len(files)):
        #     print("\t"+files[i])
    else:
        raise ValueError("Wrong number of input files.")
        exit(1)

    ed_found = False
    idx = 0
    while not ed_found:
        filename_ed_X = os.path.join(folder_path, folder_name + \
            '_frame'+str(idx).zfill(2)+'.nii.gz')

        if os.path.isfile(filename_ed_X):
            ed_found=True
            print("ED Found")
            print('\t' + filename_ed_X)
        idx += 1

    es_found = False
    while not es_found:
        filename_es_X = os.path.join(folder_path, folder_name + \
            '_frame'+str(idx).zfill(2)+'.nii.gz')

        if os.path.isfile(filename_es_X):
            es_found=True
            print("ES Found")
            print('\t' + filename_es_X)
        idx += 1

    
    reader = sitk.ImageFileReader()
    
    reader.SetFileName(filename_ed_X)
    image = reader.Execute()
    nda = sitk.GetArrayFromImage(image)
    # print(nda.shape)
    # for k in reader.GetMetaDataKeys():
    #   v = reader.GetMetaData(k)
    #   print("({0}) = = \"{1}\"".format(k,v))
    filename = folder_name + '_ED.hdf5'
    saveToHdf5(nda, dataset_names[2], ev_output_path, filename)


    reader.SetFileName(filename_es_X)
    image = reader.Execute()
    nda = sitk.GetArrayFromImage(image)
    # print(nda.shape)
    # for k in reader.GetMetaDataKeys():
    #   v = reader.GetMetaData(k)
    #   print("({0}) = = \"{1}\"".format(k,v))
    filename = folder_name + '_ES.hdf5'
    saveToHdf5(nda, dataset_names[2], ev_output_path, filename)
