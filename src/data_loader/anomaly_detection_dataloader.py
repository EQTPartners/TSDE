import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


root_path = './data/anomaly_detection/'
class PSMSegLoader(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing the PSM dataset for anomaly detection.
    The dataset is segmented based on window size and step (for sliding).

    Parameters:
    - win_size: The size of the window to segment the dataset.
    - step: The step size for sliding window.
    - flag: Indicates the dataset split to use ('train', 'val', 'test').

    Attributes:
    - train: Training data after scaling.
    - val: Validation data after scaling.
    - test: Test data after scaling.
    - test_labels: Labels for the test data.
    """
    def __init__(self, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.root_path = root_path+'PSM/'
        data = pd.read_csv(os.path.join(self.root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(self.root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(self.root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            data_point, label = np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            data_point, label = np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            data_point, label = np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            data_point, label = np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        s = {
            'observed_data': data_point,
            'observed_mask': np.ones_like(data_point),
            'gt_mask': np.ones_like(data_point),
            'timepoints': np.arange(self.win_size) * 1.0, 
            'feature_id': np.arange(data_point.shape[1]) * 1.0, 
            'label': label,
        }

        return s

class MSLSegLoader(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing the MSL dataset for anomaly detection.
    The dataset is segmented based on window size and step (for sliding).

    Parameters:
    - win_size: The size of the window to segment the dataset.
    - step: The step size for sliding window.
    - flag: Indicates the dataset split to use ('train', 'val', 'test').

    Attributes:
    - train: Training data after scaling.
    - val: Validation data after scaling.
    - test: Test data after scaling.
    - test_labels: Labels for the test data.
    """
    def __init__(self, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.root_path = root_path+'MSL/'
        self.scaler = StandardScaler()
        data = np.load(os.path.join(self.root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(self.root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(self.root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            data_point, label = np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            data_point, label = np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            data_point, label = np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            data_point = np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            
        s = {
            'observed_data': data_point,
            'observed_mask': np.ones_like(data_point),
            'gt_mask': np.ones_like(data_point),
            'timepoints': np.arange(self.win_size) * 1.0, 
            'feature_id': np.arange(data_point.shape[1]) * 1.0, 
            'label': label,
        }

        return s


class SMAPSegLoader(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing the SMAP dataset for anomaly detection.
    The dataset is segmented based on window size and step (for sliding).

    Parameters:
    - win_size: The size of the window to segment the dataset.
    - step: The step size for sliding window.
    - flag: Indicates the dataset split to use ('train', 'val', 'test').

    Attributes:
    - train: Training data after scaling.
    - val: Validation data after scaling.
    - test: Test data after scaling.
    - test_labels: Labels for the test data.
    """
    def __init__(self, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.root_path = root_path+'SMAP/'
        self.scaler = StandardScaler()
        data = np.load(os.path.join(self.root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(self.root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(self.root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            data_point, label = np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            data_point, label = np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            data_point, label = np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            data_point, label = np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        
        s = {
            'observed_data': data_point,
            'observed_mask': np.ones_like(data_point),
            'gt_mask': np.ones_like(data_point),
            'timepoints': np.arange(self.win_size) * 1.0, 
            'feature_id': np.arange(data_point.shape[1]) * 1.0, 
            'label': label,
        }

        return s


class SMDSegLoader(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing the SMD dataset for anomaly detection.
    The dataset is segmented based on window size and step (for sliding).

    Parameters:
    - win_size: The size of the window to segment the dataset.
    - step: The step size for sliding window.
    - flag: Indicates the dataset split to use ('train', 'val', 'test').

    Attributes:
    - train: Training data after scaling.
    - val: Validation data after scaling.
    - test: Test data after scaling.
    - test_labels: Labels for the test data.
    """
    def __init__(self, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.root_path = root_path+'SMD/'
        self.scaler = StandardScaler()
        data = np.load(os.path.join(self.root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(self.root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(self.root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            data_point, label = np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            data_point, label = np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            data_point, label = np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            data_point, label = np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        
        s = {
            'observed_data': data_point,
            'observed_mask': np.ones_like(data_point),
            'gt_mask': np.ones_like(data_point),
            'timepoints': np.arange(self.win_size) * 1.0, 
            'feature_id': np.arange(data_point.shape[1]) * 1.0, 
            'label': label,
        }

        return s


class SWATSegLoader(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing the SWaT dataset for anomaly detection.
    The dataset is segmented based on window size and step (for sliding).

    Parameters:
    - win_size: The size of the window to segment the dataset.
    - step: The step size for sliding window.
    - flag: Indicates the dataset split to use ('train', 'val', 'test').

    Attributes:
    - train: Training data after scaling.
    - val: Validation data after scaling.
    - test: Test data after scaling.
    - test_labels: Labels for the test data.
    """
    def __init__(self, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.root_path = root_path+'SWaT/'
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(self.root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(self.root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            data_point, label = np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            data_point, label = np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            data_point, label = np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            data_point, label = np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        
        s = {
            'observed_data': data_point,
            'observed_mask': np.ones_like(data_point),
            'gt_mask': np.ones_like(data_point),
            'timepoints': np.arange(self.win_size) * 1.0, 
            'feature_id': np.arange(data_point.shape[1]) * 1.0, 
            'label': label,
        }

        return s

        
def anomaly_detection_dataloader(dataset_name, win_size=100, batch_size=128):
    """
    Creates data loaders for training, validation, and testing datasets for anomaly detection.

    Parameters:
    - dataset_name: The name of the dataset to use ('SMAP', 'PSM', 'MSL', 'SMD', 'SWAT').
    - win_size: The window size for segmenting the dataset.
    - batch_size: The size of the batch for data loading.

    Returns:
    - A tuple of DataLoader objects for the training, validation, and testing datasets.
    """

    if dataset_name == "SMAP":
        Dataset = SMAPSegLoader
    elif dataset_name == "PSM":
        Dataset = PSMSegLoader
    elif dataset_name == "MSL":
        Dataset = MSLSegLoader
    elif dataset_name == "SMD":
        Dataset = SMDSegLoader
    elif dataset_name == "SWAT":
        Dataset = SWATSegLoader
    
    train_dataset = Dataset(win_size=win_size, flag='train') 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    valid_dataset = Dataset(win_size=win_size, flag='val') 
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    test_dataset = Dataset(win_size=win_size, flag='test') 
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    print(len(train_dataset))
    
    return train_dataloader, valid_dataloader, test_dataloader