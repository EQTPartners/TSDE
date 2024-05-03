import numpy as np
import os
import datetime
import json
import pickle
import torch
from torch.utils.data import DataLoader, Dataset


def preprocess_dataset(dataset_name, train_length, test_length, skip_length, history_length):
    """
    Preprocesses and saves a specified dataset for forecasting task, including training, testing, and validation sets.
    This function saves the corresponding processed splits (train and test), and the mean and standard deviation for the training set used for normalization.

    
    Parameters:
    - dataset_name (str): The name of the dataset. It should be one of the following: electricity, solar, taxi, traffic or wiki.
    - train_length (int): The length of the training sequences. It refers to the total number of timestamps in  training and validation sets.
    - test_length (int): The length of the testing sequences. It refers to the total number of timestamps in the test set.
    - skip_length (int): The number of sequences to skip between training and testing. The number of timestamps to skip in the test set (we are evaluating only on a subset).
    - history_length (int): The length of the historical data to consider. The total number of timestamps to use as history window in every MTS.


    """

    path_train = f'./data/{dataset_name}/{dataset_name}_nips/train/data.json' #train
    path_test = f'./data/{dataset_name}/{dataset_name}_nips/test/data.json' #test
    

    main_data=[]
    mask_data=[]
    
    hour_data=None
    

    with open(path_train, 'r') as file:
        data_train = [json.loads(line) for line in file]

    with open(path_test, 'r') as file:
        data_test = [json.loads(line) for line in file]

    ## Prepare Train Sequences
    for obj in data_train:
        tmp_data = np.array(obj['target'])
        tmp_mask = np.ones_like(tmp_data)

        if len(tmp_data) == train_length and hour_data is None:
            c_time = datetime.datetime.strptime(obj['start'],'%Y-%m-%d %H:%M:%S')
            hour_data = []
            day_data = []
            time_data = []
            for k in range(train_length):
                time_data.append(c_time)
                hour_data.append(c_time.hour)
                day_data.append(c_time.weekday())
                c_time = c_time + datetime.timedelta(hours=1)
        else:
            if len(tmp_data) != train_length: #fill NA by 0 
                tmp_padding = np.zeros(train_length-len(tmp_data))
                tmp_data = np.concatenate([tmp_padding,tmp_data])
                tmp_mask = np.concatenate([tmp_padding,tmp_mask])

        
        main_data.append(tmp_data)
        mask_data.append(tmp_mask)

    ## Prepare Test Sequences
    if dataset_name != "solar":
        cnt = 0
        ind = 0

        for line in data_test:
            cnt+=1
            if cnt <=skip_length:
                continue
            tmp_data = np.array(line['target'])
            tmp_data = tmp_data[-test_length-history_length:] 

            tmp_mask = np.ones_like(tmp_data)

            main_data[ind] = np.concatenate([main_data[ind],tmp_data])
            mask_data[ind] = np.concatenate([mask_data[ind],tmp_mask])
            c_time = datetime.datetime.strptime(obj['start'],'%Y-%m-%d %H:%M:%S')
            for i in range(test_length+history_length):
                time_data.append(c_time)
                hour_data.append(c_time.hour)
                day_data.append(c_time.weekday())
                c_time = c_time + datetime.timedelta(hours=1)
            ind += 1
    main_data = np.stack(main_data,-1)
    mask_data = np.stack(mask_data,-1)
    print('Main data shape', main_data.shape)
    ## Save means
    mean_data = main_data[:-test_length-history_length].mean(0)
    std_data = main_data[:-test_length-history_length].std(0)


    ## Save means
    paths=f'./data/{dataset_name}/{dataset_name}_nips/meanstd.pkl' 
    if os.path.isfile(paths) == False:
        with open(paths, 'wb') as f:
            pickle.dump([mean_data,std_data],f)
            
    ## Save sequences
    paths=f'./data/{dataset_name}/{dataset_name}_nips/data.pkl' 
    if os.path.isfile(paths) == False:
        with open(paths, 'wb') as f:
            pickle.dump([main_data,mask_data],f)


def preprocess_taxi(train_length, test_length, skip_length, history_length):
    """
    Specialized preprocessing function for the taxi dataset, similar to preprocess_dataset but tailored to its structure. 
    
    Parameters:
    - train_length (int): The length of the training sequences. It refers to the total number of timestamps in  training and validation sets.
    - test_length (int): The length of the testing sequences. It refers to the total number of timestamps in the test set.
    - skip_length (int): The number of sequences to skip between training and testing. The number of timestamps to skip in the test set (we are evaluating only on a subset).
    - history_length (int): The length of the historical data to consider. The total number of timestamps to use as history window in every MTS.
    """

    path_train = f'./data/taxi/taxi_nips/train/data.json' #train
    path_test = f'./data/taxi/taxi_nips/test/data.json' #test
    
    
    main_data=[]
    mask_data=[]
    
    hour_data=None
    

    with open(path_train, 'r') as file:
        data_train = [json.loads(line) for line in file]

    with open(path_test, 'r') as file:
        data_test = [json.loads(line) for line in file]
    
    ## Prepare Train Sequences
    for obj in data_train:
        tmp_data = np.array(obj['target'])
        tmp_mask = np.ones_like(tmp_data)

        if len(tmp_data) == train_length and hour_data is None:
            c_time = datetime.datetime.strptime(obj['start'],'%Y-%m-%d %H:%M:%S')
            hour_data = []
            day_data = []
            time_data = []
            for k in range(train_length):
                time_data.append(c_time)
                hour_data.append(int(c_time.hour+c_time.minute/30))
                day_data.append(c_time.weekday())
                c_time = c_time + datetime.timedelta(minutes=30)
        else:
            if len(tmp_data) != train_length: #fill NA by 0 
                tmp_padding = np.zeros(train_length-len(tmp_data))
                tmp_data = np.concatenate([tmp_padding,tmp_data])
                tmp_mask = np.concatenate([tmp_padding,tmp_mask])

        
        main_data.append(tmp_data)
        mask_data.append(tmp_mask)

    ## Prepare Test Sequences
    cnt = 0
    ind = 0

    for line in data_test:
        cnt+=1
        if cnt <=skip_length:
            continue
        tmp_data = np.array(line['target'])
        tmp_data = tmp_data[-test_length-history_length:] 
        
        tmp_mask = np.ones_like(tmp_data)
        
        main_data[ind] = np.concatenate([main_data[ind],tmp_data])
        mask_data[ind] = np.concatenate([mask_data[ind],tmp_mask])
        c_time = datetime.datetime.strptime(obj['start'],'%Y-%m-%d %H:%M:%S')
        for i in range(test_length+history_length):
            time_data.append(c_time)
            hour_data.append(c_time.hour)
            day_data.append(c_time.weekday())
            c_time = c_time + datetime.timedelta(minutes=30)
        ind += 1
    
    main_data = np.stack(main_data,-1)
    mask_data = np.stack(mask_data,-1)
    

    ## Save mean
    mean_data = main_data[:-test_length-history_length].mean(0)
    std_data = main_data[:-test_length-history_length].std(0)

    ## Save means
    paths=f'./data/taxi/taxi_nips/meanstd.pkl' 
    if os.path.isfile(paths) == False:
        with open(paths, 'wb') as f:
            pickle.dump([mean_data,std_data],f)

    ## Save sequences
    paths=f'./data/taxi/taxi_nips/data.pkl' 
    if os.path.isfile(paths) == False:
        with open(paths, 'wb') as f:
            pickle.dump([main_data,mask_data],f)
    
def preprocess_wiki(train_length, test_length, skip_length, history_length):
    """
    Specialized preprocessing function for the wiki dataset, similar to preprocess_dataset but tailored to its structure.
    
    Parameters:
    - train_length (int): The length of the training sequences. It refers to the total number of timestamps in  training and validation sets.
    - test_length (int): The length of the testing sequences. It refers to the total number of timestamps in the test set.
    - skip_length (int): The number of sequences to skip between training and testing. The number of timestamps to skip in the test set (we are evaluating only on a subset).
    - history_length (int): The length of the historical data to consider. The total number of timestamps to use as history window in every MTS.
    """
    path_train = f'./data/wiki/wiki_nips/train/data.json' #train
    path_test = f'./data/wiki/wiki_nips/test/data.json' #test
    
    
    main_data=[]
    mask_data=[]
    
    hour_data=None
    

    with open(path_train, 'r') as file:
        data_train = [json.loads(line) for line in file]

    with open(path_test, 'r') as file:
        data_test = [json.loads(line) for line in file]

    ## Prepare Train Sequences
    for obj in data_train:
        tmp_data = np.array(obj['target'])
        tmp_mask = np.ones_like(tmp_data)

        if len(tmp_data) == train_length and hour_data is None:
            c_time = datetime.datetime.strptime(obj['start'],'%Y-%m-%d %H:%M:%S')
            hour_data = []
            day_data = []
            time_data = []
            for k in range(train_length):
                time_data.append(c_time)
                hour_data.append(c_time.hour)
                day_data.append(c_time.weekday())
                c_time = c_time + datetime.timedelta(days=1)
        else:
            if len(tmp_data) != train_length: #fill NA by 0 
                tmp_padding = np.zeros(train_length-len(tmp_data))
                tmp_data = np.concatenate([tmp_padding,tmp_data])
                tmp_mask = np.concatenate([tmp_padding,tmp_mask])

        
        main_data.append(tmp_data)
        mask_data.append(tmp_mask)

    ## Prepare Test Sequences
    cnt = 0
    ind = 0

    for line in data_test:
        cnt+=1
        if cnt <=skip_length:
            continue
        tmp_data = np.array(line['target'])
        tmp_data = tmp_data[-test_length-history_length:] 
        
        tmp_mask = np.ones_like(tmp_data)
        
        main_data[ind] = np.concatenate([main_data[ind],tmp_data])
        mask_data[ind] = np.concatenate([mask_data[ind],tmp_mask])
        c_time = datetime.datetime.strptime(obj['start'],'%Y-%m-%d %H:%M:%S')
        
        ind += 1
    main_data = np.stack(main_data[-2000:],-1)
    mask_data = np.stack(mask_data[-2000:],-1)
    
    mean_data = main_data[:-test_length-history_length].mean(0)
    std_data = main_data[:-test_length-history_length].std(0)

    ## Save means
    paths=f'./data/wiki/wiki_nips/meanstd.pkl' 
    if os.path.isfile(paths) == False:
        with open(paths, 'wb') as f:
            pickle.dump([mean_data,std_data],f)

    ## Save sequences
    paths=f'./data/wiki/wiki_nips/data.pkl' 
    if os.path.isfile(paths) == False:
        with open(paths, 'wb') as f:
            pickle.dump([main_data,mask_data],f)

class Forecasting_Dataset(Dataset):
    """
    A PyTorch Dataset class for loading and preparing forecasting data.
    
    Parameters:
    - dataset_name (str): The name of the dataset. One of the following: electricity, solar, traffic, taxi or wiki.
    - train_length, skip_length, valid_length, test_length, pred_length, history_length (int): Parameters defining the dataset structure and lengths of different segments as described in the processing functions.
    - is_train (int): Indicator of the dataset split (0 for test, 1 for train, 2 for valid).
    """
    def __init__(self,  dataset_name, train_length, skip_length, valid_length, test_length, pred_length, history_length, is_train):
        self.history_length = history_length
        self.pred_length = pred_length
        self.test_length = test_length
        self.valid_length = valid_length
        self.data_type = dataset_name
        self.seq_length = self.pred_length+self.history_length

        if dataset_name == 'taxi':
            preprocess_taxi(train_length, test_length, skip_length, history_length)
        elif dataset_name == 'wiki':
            preprocess_wiki(train_length, test_length, skip_length, history_length)
        else:
            preprocess_dataset(dataset_name, train_length, test_length, skip_length, history_length)

        paths = f'./data/{self.data_type}/{self.data_type}_nips/data.pkl' 
        mean_path = f'./data/{self.data_type}/{self.data_type}_nips/meanstd.pkl'
        with open(paths, 'rb') as f:
            self.main_data,self.mask_data=pickle.load(f)
        with open(mean_path, 'rb') as f:
            self.mean_data,self.std_data=pickle.load(f)
            
        self.main_data = (self.main_data - self.mean_data) / np.maximum(1e-5,self.std_data)

        data_length = len(self.main_data)
        if is_train == 0: #test
            start = data_length - self.seq_length - self.test_length + self.pred_length
            end = data_length - self.seq_length + self.pred_length
            self.use_index = np.arange(start,end,self.pred_length)
            print('Test', start, end)
            
        if is_train == 2: #valid 
            start = data_length - self.seq_length - self.valid_length - self.test_length + self.pred_length
            end = data_length - self.seq_length - self.test_length + self.pred_length
            self.use_index = np.arange(start,end,self.pred_length)
            print('Val', start, end)
        if is_train == 1:
            start = 0
            end = data_length - self.seq_length - self.valid_length - self.test_length + 1
            self.use_index = np.arange(start,end,1)
            print('Train', start, end)
        
        
    def __getitem__(self, orgindex):
        """
        Gets the MTS at the specified index.
        
        Parameters:
        - orgindex (int): The index of the MTS (index of the start timestamp of the sequence).
        
        Returns:
        - dict: A dictionary containing 'observed_data', 'observed_mask', 'gt_mask', 'timepoints', and 'feature_id'.
        """
        index = self.use_index[orgindex]
        target_mask = self.mask_data[index:index+self.seq_length].copy()
        target_mask[-self.pred_length:] = 0. 
        s = {
            'observed_data': self.main_data[index:index+self.seq_length],
            'observed_mask': self.mask_data[index:index+self.seq_length],
            'gt_mask': target_mask,
            'timepoints': np.arange(self.seq_length) * 1.0, 
            'feature_id': np.arange(self.main_data.shape[1]) * 1.0, 
        }

        return s
    
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
        - int: The total number of samples.
        """
        return len(self.use_index)
    

def get_dataloader_forecasting(dataset_name, train_length, skip_length, valid_length=24*5, test_length =24*7, pred_length=24, history_length=168, batch_size=8, device='cuda:0'):
        """
        Prepares DataLoader objects for the forecasting datasets.
        
        Parameters:
        - dataset_name (str): The name of the dataset.
        - train_length, skip_length, valid_length, test_length, pred_length, history_length, batch_size (int): Various parameters defining dataset and DataLoader configurations.
        - device (str): The device to use for loading tensors.
        
        Returns:
        - Tuple[DataLoader, DataLoader, DataLoader, Tensor, Tensor]: Training, validation, and testing DataLoaders, along with scale and mean scale tensors used for normalization.
        """

        train_dataset = Forecasting_Dataset(dataset_name, train_length, skip_length, valid_length, test_length, pred_length, history_length, is_train=1)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        
        valid_dataset = Forecasting_Dataset(dataset_name, train_length, skip_length, valid_length, test_length, pred_length, history_length, is_train=2)
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False)
        
        test_dataset = Forecasting_Dataset(dataset_name, train_length, skip_length, valid_length, test_length, pred_length, history_length, is_train=0)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)
        scaler = torch.from_numpy(train_dataset.std_data).to(device).float()
        mean_scaler = torch.from_numpy(train_dataset.mean_data).to(device).float()
        return train_loader, valid_loader, test_loader, scaler, mean_scaler