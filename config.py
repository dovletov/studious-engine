import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'DATA')
TRAIN_DIR = os.path.join(DATA_DIR, 'training')
TEST_DIR = os.path.join(DATA_DIR, 'testing')

NP_DIR = os.path.join(BASE_DIR, 'NP_DATA')
NP_TRAIN_DIR = os.path.join(NP_DIR, 'training') 
NP_TEST_DIR = os.path.join(NP_DIR, 'testing')

RESULTS_DIR = os.path.join(BASE_DIR, 'RESULTS')
