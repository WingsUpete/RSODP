import torch

# Basic
LEARNING_RATE_DEFAULT = 1e-2    # 0.01
MAX_EPOCHS_DEFAULT = 200
EVAL_FREQ_DEFAULT = 5
BATCH_SIZE_DEFAULT = 20
OPTIMIZER_DEFAULT = 'ADAM'
WEIGHT_DECAY_DEFAULT = 0.01
DATA_DIR_DEFAULT = 'data/ny2016_0101to0331/'
LOG_DIR_DEFAULT = 'log/'
WORKERS_DEFAULT = 4
USE_GPU_DEFAULT = 1
NETWORK_DEFAULT = 'Gallat'
MODE_DEFAULT = 'train'
EVAL_DEFAULT = 'eval.pt'   # should be a model file name
MODEL_SAVE_DIR_DEFAULT = 'model_save/'

MAX_NORM_DEFAULT = 10.0
FEAT_DIM_DEFAULT = 13
QUERY_DIM_DEFAULT = 11
HIDDEN_DIM_DEFAULT = 16
SCALE_FACTOR_DEFAULT_D = 16000
SCALE_FACTOR_DEFAULT_G = 7000

NUM_HEADS_DEFAULT = 3

HISTORICAL_RECORDS_NUM_DEFAULT = 7
TIME_SLOT_ENDURANCE_DEFAULT = 1     # hour

TEMP_FEAT_NAMES = ['St', 'Sp', 'Stpm', 'Stpp']

D_PERCENTAGE_DEFAULT = 0.8
G_PERCENTAGE_DEFAULT = 0.2

TRAIN_TYPES = ['normal', 'pretrain', 'retrain']
TRAIN_TYPE_DEFAULT = 'normal'

RETRAIN_MODEL_PATH_DEFAULT = 'res/Gallat_pretrain/20210514_07_17_13.pth'

DATA_TOTAL_H = -1
DATA_START_H = -1

# Debug
DEBUG = True

# Single Value Tensor for Metrics Threshold (0, 3, 5)
ZERO_TENSOR = torch.Tensor([0])
METRICS_THRESHOLD_DEFAULT = ZERO_TENSOR

# Customize: DIY
EVAL_METRICS_THRESHOLD_SET = [0, 3, 5]
