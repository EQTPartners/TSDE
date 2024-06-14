import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler


'''class StandardScaler(object):

    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = self.mean
        std = self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = self.mean
        std = self.std
        return (data * std) + mean'''


class Dataset_Custom(Dataset):

    def __init__(self, root_path='dataset/electricity', flag='train', size=None, features='M', data_path='electricity.csv', target='OT', scale=True,
                 inverse=False, timeenc=1, freq='t', cols=None, percentage=1):
        # size [seq_len, label_len, pred_len]
        # info
        super().__init__()
        self.seq_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.percentage = percentage
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        length = len(df_raw)*self.percentage
        num_train = int(length*0.7)
        num_test = int(length*0.2)
        num_vali = int(length*0.1)
        # num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, num_train+num_vali-self.seq_len]
        border2s = [num_train, num_train+num_vali, num_train+num_vali+num_test]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features == 'M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            
            '''for i in range(len(self.scaler.std)):
                if self.scaler.std[i] == 0:
                    print(i)'''
            # print(len(self.scaler.std))
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        # df_stamp = df_raw[['date']][border1:border2]
        # df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_stamp = pd.date_range(start='4/1/2018',periods=border2-border1, freq='H')

        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]

    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:r_end]
        
        observed_mask = np.ones_like(seq_x)
        target_mask=observed_mask.copy()
        target_mask[-self.pred_len:] = 0
        s = {
            'observed_data': seq_x,
            'observed_mask': observed_mask,
            'gt_mask': target_mask,
            'timepoints': np.arange(self.seq_len+self.pred_len) * 1.0, 
            'feature_id': np.arange(seq_x.shape[1]) * 1.0, 
        }
        #return seq_x, seq_y, seq_x_mark, seq_y_mark
        return s

        
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    

def get_dataloader_elec(pred_length=96, history_length=192, batch_size=8, device='cuda:0'):


        train_dataset = Dataset_Custom(flag='train', size=[history_length, pred_length])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        valid_dataset = Dataset_Custom(flag='val', size=[history_length, pred_length])
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
        test_dataset = Dataset_Custom(flag='test', size=[history_length, pred_length])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print('Lengths', len(train_dataset), len(valid_dataset), len(test_dataset))
       
        return train_loader, valid_loader, test_loader, test_dataset


def get_dataloader_traffic(pred_length=96, history_length=192, batch_size=8, device='cuda:0'):

        train_dataset = Dataset_Custom(root_path='dataset/traffic', data_path='traffic.csv', flag='train', size=[history_length, pred_length])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        valid_dataset = Dataset_Custom(root_path='dataset/traffic', data_path='traffic.csv', flag='val', size=[history_length, pred_length])
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
        test_dataset = Dataset_Custom(root_path='dataset/traffic', data_path='traffic.csv', flag='test', size=[history_length, pred_length])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print('Lengths', len(train_dataset), len(valid_dataset), len(test_dataset))
       
        return train_loader, valid_loader, test_loader, test_dataset
