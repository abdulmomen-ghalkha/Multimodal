import torchvision.transforms as transf
import pandas as pd




def read_drone_drone(distribution):
    if distribution == "IID":
        dataset_dir = "feature_IID"
    elif distribution == "NIID":
        dataset_dir = "feature_NIID"
    else:
        raise ValueError("Invalid dataset distribution.")