from torch.utils.data import Dataset
import numpy as np
import torch


class DynamicalSystem(Dataset):
    def __init__(self, mode, system, norm=True, horizon=50):
        """Dataset:
                    - mode : {train, val, test} choose the split to load
                    - system : {lorenz, vanderpol, lotkavolterra, meanfield}
                    - norm : normalize data so that the train split lies between -1 and 1
                    - horizon : length of the trajectory"""
        super(DynamicalSystem, self).__init__()

        assert mode in ['train', 'val', 'test'], "Invalid mode, please choose between {train,val,test}"
        assert system in ['vanderpol', 'lorenz', 'lotkavolterra',
                          'meanfield'], "Invalid system, please choose between {vanderpol, lorenz, lotkavolterra, meanfield}"
        try:
            self.data = np.load("Dataset/" + system + "/" + mode + ".npy")
        except FileNotFoundError:
            self.data = np.load("../Dataset/" + system + "/" + mode + ".npy")

        self.data = self.data[..., :horizon]  # Crop trajectories to the horizon

        self.dim = 2 if system in ["vanderpol", "lotkavolterra"] else 3
        if system == "vanderpol":
            self.dt = 0.25  # Timestep
            mini, maxi = -5.126374043286566, 5.11807001494796
        elif system == "lorenz":
            self.dt = 0.02
            mini, maxi = -27.933126229196468, 29.252984670297696
        elif system == "lotkavolterra":
            self.dim = 2
            mini, maxi = 0.0019810551671188836, 4.9303848543252045
        elif system == "meanfield":
            self.dt = 0.05
            mini, maxi = -1.0887941375684436, 1.0799357395199474

        if norm:
            self.data = ((maxi - self.data) / (maxi - mini)) * 2 - 1

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, item):
        return torch.Tensor(self.data[item])


if __name__ == '__main__':
    d = DynamicalSystem("test", "vanderpol")
