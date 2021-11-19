import torch

# DEBUG FLAGS
TRAIN_JUST_ONE_ROUND = False
PROFILE = False
CHECK_GRADS = False

# Basic
LEARNING_RATE_DEFAULT = 1e-2    # 0.01
MAX_EPOCHS_DEFAULT = 300
EVAL_FREQ_DEFAULT = 5
BATCH_SIZE_DEFAULT = 32
OPTIMIZER_DEFAULT = 'ADAM'
WEIGHT_DECAY_DEFAULT = 0.01
DATA_DIR_DEFAULT = 'data/ny2016_0101to0331/'
LOG_DIR_DEFAULT = 'log/'
WORKERS_DEFAULT = 36
USE_GPU_DEFAULT = 1
NETWORK_DEFAULT = 'GallatExt'
MODE_DEFAULT = 'train'
EVAL_DEFAULT = 'eval.pt'   # should be a model file name
MODEL_SAVE_DIR_DEFAULT = 'model_save/'

MAX_NORM_DEFAULT = 10.0
FEAT_DIM_DEFAULT = 35
QUERY_DIM_DEFAULT = 33
HIDDEN_DIM_DEFAULT = 16

LOSS_FUNC_DEFAULT = 'MSELoss'

NUM_HEADS_DEFAULT = 3

HISTORICAL_RECORDS_NUM_DEFAULT = 7
TIME_SLOT_ENDURANCE_DEFAULT = 1     # hour

TUNE_DEFAULT = 1

TEMP_FEAT_NAMES = ['St', 'Sp', 'Stpm', 'Stpp']
HA_FEAT_DEFAULT = 'all'     # ['all', 'tendency', 'periodicity']

D_PERCENTAGE_DEFAULT = 0.8
G_PERCENTAGE_DEFAULT = 1 - D_PERCENTAGE_DEFAULT

REF_EXTENT = 0.2    # If -1, using scaling factor scheme; Else, a simple leverage

TRAIN_TYPES = ['normal', 'pretrain', 'retrain']
TRAIN_TYPE_DEFAULT = 'normal'

RETRAIN_MODEL_PATH_DEFAULT = 'res/Gallat_pretrain/20210514_07_17_13.pth'

DATA_TOTAL_H = -1
DATA_START_H = -1

# Customize: DIY
EVAL_METRICS_THRESHOLD_SET = [0, 3, 5]

METRICS_FOR_WHAT = ['Demand', 'OD']

