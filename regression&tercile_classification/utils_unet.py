import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import r2_score, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.samples = data_x
        self.outputs = data_y

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.outputs[idx]
    

# new verison for larger grid
def get_index_by_lat_lon(lat, lon):
    i_lat = int((52.75-lat)/0.5)  #  We are doing descending order for lat
    i_lon = int((lon-233.25)/0.5)
    return i_lat,i_lon


def get_lat_lon_by_index(i_lat,i_lon):
    lat = 52.75 - 0.5*i_lat
    lon = 233.25 + 0.5*i_lon
    return lat,lon


def pred(model, dataloader_train, dataloader_test, US_mask_tensor, device):
    pred_train = []
    pred_test = []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader_train: 
            out = model(x.to(device).float())
            out = out * US_mask_tensor.to(device)
            pred_train.append(out.cpu().numpy())

        for x, y in dataloader_test: 
            out = model(x.to(device).float())
            out = out * US_mask_tensor.to(device)
            pred_test.append(out.cpu().numpy())

    pred_train = np.vstack(pred_train)
    pred_test = np.vstack(pred_test)
    pred_all = np.concatenate((pred_train, pred_test), axis=0)
    
    return pred_all


def crop_to_US_land(img): 
    coords = np.argwhere(img)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    cropped = img[x_min:x_max+1, y_min:y_max+1]
    return cropped

