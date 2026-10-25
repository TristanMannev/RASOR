import torch

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

# mlflow
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

# Seed for reproductability
SEED = 42

# Data Path
# GOWT video
TRAIN_DATA_GOWT = "data/Fluo-N2DH-GOWT1_train/01"
TRAIN_SEG_DATA_GOWT = "data/Fluo-N2DH-GOWT1_train/01_ERR_SEG_TRA"
TEST_DATA_GOWT = "data/Fluo-N2DH-GOWT1_train/02"
TEST_SEG_DATA_GOWT = "data/Fluo-N2DH-GOWT1_train/02_ERR_SEG_TRA"
# HeLa video
TRAIN_DATA_HELA = "data/Fluo-N2DL-HeLa_train/01"
TRAIN_SEG_DATA_HELA = "data/Fluo-N2DL-HeLa_train/01_ERR_SEG_TRA"
TEST_DATA_HELA = "data/Fluo-N2DL-HeLa_train/02"
TEST_SEG_DATA_HELA = "data/Fluo-N2DL-HeLa_train/02_ERR_SEG_TRA"
# hsc video
TRAIN_DATA_HSC = "data/BF-C2DL-HSC/01"
TRAIN_SEG_DATA_HSC = "data/BF-C2DL-HSC/01_ERR_SEG_TRA"
TEST_DATA_HSC = "data/BF-C2DL-HSC/02"
TEST_SEG_DATA_HSC = "data/BF-C2DL-HSC/02_ERR_SEG_TRA"
# musc video
TRAIN_DATA_MUSC = "data/BF-C2DL-MuSC/01"
TRAIN_SEG_DATA_MUSC = "data/BF-C2DL-MuSC/01_ERR_SEG_TRA"
TEST_DATA_MUSC = "data/BF-C2DL-MuSC/02"
TEST_SEG_DATA_MUSC = "data/BF-C2DL-MuSC/02_ERR_SEG_TRA"
# huh video
TRAIN_DATA_HUH = "data/Fluo-C2DL-Huh7/01"
TEST_DATA_HUH = "data/Fluo-C2DL-Huh7/02"
# u373 video
TRAIN_DATA_U373 = "data/PhC-C2DH-U373/01"
TRAIN_SEG_DATA_U373 = "data/PhC-C2DH-U373/01_ERR_SEG_TRA"
TEST_DATA_U373 = "data/PhC-C2DH-U373/02"
TEST_SEG_DATA_U373 = "data/PhC-C2DH-U373/02_ERR_SEG_TRA"


# Constants
HEIGHT = 1024
WIDTH = 1024
DEPTH = 32
SHAPE = [HEIGHT, WIDTH]
