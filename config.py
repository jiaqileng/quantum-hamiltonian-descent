import os

MAIN_DIR = "/Users/lengjiaqi/QHD_DATA"

if MAIN_DIR is None:
    assert False, "Please set MAIN_DIR to point to the QHD Data directory"

DATA_DIR_2D = os.path.join(MAIN_DIR, "NonCVX-2d")
DATA_DIR_QP = os.path.join(MAIN_DIR, "QP")
