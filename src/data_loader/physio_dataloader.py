import pickle
import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# 35 attributes which contains enough non-values
attributes = ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose', 'SaO2',
              'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS',
              'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP',
              'Creatinine', 'ALP']


def extract_hour(x):
    """
    Extracts and returns the hour from a time string. Process PhysioNet data at an hourly granularity.

    Parameters:
    - x (str): Time string in the format 'HH:MM'.

    Returns:
    - int: The hour part extracted from the time string.

    """
    h, _ = map(int, x.split(":"))
    return h


def parse_data(x):
    """
    Processes a pandas DataFrame to extract the recorded value for each attribute in 'attributes'.
    Returns a list of values with NaN for any missing attribute.

    Parameters:
    - x (DataFrame): DataFrame containing time series data for various attributes.
    
    Returns:
    - list: A list of observed values for the specified attributes, with NaN for missing ones.
    """

    # extract the last value for each attribute
    x = x.set_index("Parameter").to_dict()["Value"]

    values = []

    for attr in attributes:
        if x.__contains__(attr):
            values.append(x[attr])
        else:
            values.append(np.nan)
    return values

def extract_record_id_and_death_status_as_dict(file_path):
    """
    Creates a dictionary mapping patient RecordID to their in-hospital death status from a CSV file.
    
    Parameters:
    - file_path (str): Path to the CSV file containing RecordID and In-hospital_death columns.
    
    Returns:
    - dict: A dictionary with RecordID as keys and in-hospital death status as values.
    """

    # Read the data from the file
    df = pd.read_csv(file_path)

    # Extract RecordID and In-hospital_death columns and convert to a dictionary
    result_dict = df.set_index('RecordID')['In-hospital_death'].to_dict()

    # Print or return the result as needed
    return result_dict


# File path (change this to your file path)
file_path = './data/physio/Outcomes-a.txt'

# Run the function with the file path and print the results
id_label_mapping = extract_record_id_and_death_status_as_dict(file_path)


def parse_id(id_, missing_ratio=0.1, mode='imputation'):
    """
    Reads and preprocesses patient data, applies missing data handling, and prepares it for model input.
    
    Parameters:
    - id_ (str): The patient ID to process.
    - missing_ratio (float): The ratio of data to be considered as missing, masked and used for evaluation.
    - mode (str): The mode of handling missing data, either 'imputation' or 'interpolation'.
    
    Returns:
    - Tuple containing processed data arrays and labels: (observed_values, observed_masks, gt_masks, label)
    """

    data = pd.read_csv("./data/physio/set-a/{}.txt".format(id_))
    # set hour
    data["Time"] = data["Time"].apply(lambda x: extract_hour(x))

    # create data for 48 hours x 35 attributes
    observed_values = []
    for h in range(48):
        observed_values.append(parse_data(data[data["Time"] == h]))
    observed_values = np.array(observed_values)
    observed_masks = ~np.isnan(observed_values)
    if mode == 'imputation':
        # randomly set some percentage as ground-truth
        masks = observed_masks.reshape(-1).copy()
        obs_indices = np.where(masks)[0].tolist()
        miss_indices = np.random.choice(
            obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
        )
        masks[miss_indices] = False
        gt_masks = masks.reshape(observed_masks.shape)
    elif mode == 'interpolation':
        # randomly set some percentage of timestamps values as ground-truth
        timestamps = np.arange(48)
        miss_timestamps = np.random.choice(
            timestamps, (int)(len(timestamps) * missing_ratio), replace=False
        )
        masks = observed_masks.copy()
        masks[miss_timestamps,:] = False
        gt_masks = masks
    else:
        print('Please set mode to interpolation or imputation!')

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")
    label = id_label_mapping[int(id_)]
    return observed_values, observed_masks, gt_masks, label


def get_idlist():
    """
    Scans a directory for files matching the patient ID pattern and returns a sorted list of IDs.
    
    Returns:
    - list: A sorted list of patient IDs extracted from filenames.
    """
    patient_id = []
    for filename in os.listdir("./data/physio/set-a"):
        match = re.search("\d{6}", filename)
        if match:
            patient_id.append(match.group())
    patient_id = np.sort(patient_id)
    return patient_id


class Physio_Dataset(Dataset):
    """
    A Dataset class for loading and processing PhysioNet data for use in PyTorch.
    
    Parameters:
    - eval_length (int): The length of the time series evaluation window (Total number of timestamps in each MTS(L)).
    - use_index_list (list, optional): List of indices to use from the dataset (to differentiate between training, validation and test sets).
    - missing_ratio (float): Ratio of data to mask as missing ansd use for evaluation (set to 0.1, 0.5 or 0.9).
    - seed (int): Random seed for reproducibility (used to randomly select the same subset of values and mask them).
    - mode (str): Mode for handling missing data ('imputation' or 'interpolation').
    """
    def __init__(self, eval_length=48, use_index_list=None, missing_ratio=0.0, seed=0, mode='imputation'):
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        self.labels = []
        path = (
            "./data/physio_missing" + str(missing_ratio) + "_" + mode + "_seed" + str(seed) + ".pk"
        )

        if os.path.isfile(path) == False:  # if datasetfile is none, create
            idlist = get_idlist()
            for id_ in idlist:
                try:
                    observed_values, observed_masks, gt_masks, label = parse_id(
                        id_, missing_ratio, mode=mode
                    )
                    self.observed_values.append(observed_values)
                    self.observed_masks.append(observed_masks)
                    self.gt_masks.append(gt_masks)
                    self.labels.append(label)
                except Exception as e:
                    print(id_, e)
                    continue
            self.observed_values = np.array(self.observed_values)
            self.observed_masks = np.array(self.observed_masks)
            self.gt_masks = np.array(self.gt_masks)

            # calc mean and std and normalize values
            # (it is the same normalization as Cao et al. (2018) (https://github.com/caow13/BRITS))
            tmp_values = self.observed_values.reshape(-1, 35)
            tmp_masks = self.observed_masks.reshape(-1, 35)
            mean = np.zeros(35)
            std = np.zeros(35)
            for k in range(35):
                c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
                mean[k] = c_data.mean()
                std[k] = c_data.std()
            self.observed_values = (
                (self.observed_values - mean) / std * self.observed_masks
            )

            with open(path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks, self.labels], f
                )
        else:  # load datasetfile
            with open(path, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks, self.labels = pickle.load(
                    f
                )
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        """
        Returns a sample from the dataset at the specified index.
        
        Parameters:
        - org_index (int): The index of the sample to retrieve.
        
        Returns:
        - dict: A dictionary containing the data sample.
        """
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
            "labels": self.labels[index],
        }
        return s

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
        - int: Total number of samples.
        """
        return len(self.use_index_list)


def get_dataloader_physio(seed=1, nfold=None, batch_size=16, missing_ratio=0.1, mode='imputation'):
    """
    Prepares DataLoader objects for the PhysioNet dataset for training, validation, and testing.
    
    Parameters:
    - seed (int): Random seed for reproducibility.
    - nfold (int, optional): Current fold number for cross-validation.
    - batch_size (int): Batch size for the DataLoader.
    - missing_ratio (float): Ratio of data to mask as missing.
    - mode (str): Mode for handling missing data ('imputation' or 'interpolation').
    
    Returns:
    - Tuple: Contains DataLoader objects for training, validation, and testing.
    """
    # only to obtain total length of dataset
    dataset = Physio_Dataset(missing_ratio=missing_ratio, seed=seed, mode=mode)
    indlist = np.arange(len(dataset))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    # 5-fold test
    start = (int)(nfold * 0.2 * len(dataset))
    end = (int)((nfold + 1) * 0.2 * len(dataset))
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.seed(seed)
    np.random.shuffle(remain_index)
    num_train = (int)(len(dataset) * 0.7)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    dataset = Physio_Dataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed, mode=mode
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Physio_Dataset(
        use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed, mode=mode
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Physio_Dataset(
        use_index_list=test_index, missing_ratio=missing_ratio, seed=seed, mode=mode
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, valid_loader, test_loader




class Imputed_Physio_Dataset(Dataset):
    """
    A Dataset class for loading imputed PhysioNet data in PyTorch (used for classification and clustering experiments).
    
    Parameters:
    - filename (str): Path to the file containing the imputed time series.
    - flag (str): Indicates the subset of the data ('Train', 'Val', or 'Test').
    """
    def __init__(self, filename, flag='Train'):
        
        self.file = filename
        self.flag = flag
        data_path = self.file + f'generated_outputs_{flag}_nsample100.pk'
     
        with open(data_path, 'rb') as f:
            self.samples, self.all_target, self.all_evalpoint, self.all_observed, self.all_observed_time, self.labels, self.scaler, self.mean_scaler = pickle.load(f)
            
        self.imputed_mts = self.samples.median(dim=1)[0]*(1-self.all_observed)+self.all_target*self.all_observed

    
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
        - int: Total number of samples.
        """
        return len(self.imputed_mts)
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset at the specified index.
        
        Parameters:
        - index (int): The index of the sample to retrieve.
        
        Returns:
        - dict: A dictionary containing the data sample.
        """
        s = {
            "observed_data": self.imputed_mts[index].cpu(),
            "observed_mask": np.ones_like(self.imputed_mts[index].cpu()),
            "gt_mask": np.ones_like(self.imputed_mts[index].cpu()),
            "timepoints": np.arange(self.imputed_mts[index].shape[0]),
            "y": self.labels[index],
            "labels": self.labels[index],
        }
        return s

    
def get_physio_dataloader_for_classification(filename, batch_size):
    """
    Prepares DataLoader objects for classification and clustering experiments using imputed PhysioNet data.
    
    Parameters:
    - filename (str): Path to the file containing the imputed dataset.
    - batch_size (int): Batch size for the DataLoader.
    
    Returns:
    - Tuple: Contains DataLoader objects for training, validation, and testing.
    """
    train_dataset = Imputed_Physio_Dataset(
        filename, flag = 'Train'
    )
    valid_dataset = Imputed_Physio_Dataset(
        filename, flag = 'Val'
    )
    test_dataset = Imputed_Physio_Dataset(
        filename, flag = 'Test'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=1)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    
    return train_loader, valid_loader, test_loader
    