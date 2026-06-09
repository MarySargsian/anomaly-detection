import numpy as np
import json

class TimeSeriesPreprocessor:
    def __init__(self, window_size, stride=1):
        self.window_size = window_size
        self.stride = stride
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.std(axis=0, keepdims=True) + 1e-8

    def transform(self, data):
        return (data - self.mean) / self.std

    def create_windows(self, data):
        T, C = data.shape
        windows = []
        for i in range(0, T - self.window_size + 1, self.stride):
            windows.append(data[i : i + self.window_size].T)
        return np.array(windows)

    def create_predictive_windows(self, data, pred_steps=20):
        T, C = data.shape
        x_windows = []
        y_windows = []
        for i in range(0, T - self.window_size - pred_steps + 1, self.stride):
            x_windows.append(data[i : i + self.window_size].T)
            y_windows.append(data[i + self.window_size : i + self.window_size + pred_steps].T)
        return np.array(x_windows), np.array(y_windows)

def temporal_split(data, train=0.6, val=0.2):
    n = len(data)
    t1 = int(n * train)
    t2 = int(n * (train + val))
    return data[:t1], data[t1:t2], data[t2:]
