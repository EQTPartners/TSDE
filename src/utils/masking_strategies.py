import torch
import numpy as np
import random



def imputation_mask_batch(observed_mask):
    """
    Generates a batch of masks for imputation task where certain observations are randomly masked based on a 
    sample-specific ratio.
    
    Parameters:
    - observed_mask (Tensor): A tensor indicating observed values (1 for observed, 0 for missing).
    
    Returns:
    - Tensor: A mask tensor for imputation with the same shape as `observed_mask`.
    """
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask ### array like observed mask filled with random values
    rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
    min_value, max_value = 0.1, 0.9
    for i in range(len(observed_mask)):
        sample_ratio = min_value + (max_value - min_value)*np.random.rand()  # missing ratio ## at random
        num_observed = observed_mask[i].sum().item()
        num_masked = round(num_observed * sample_ratio)
        rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
    cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
    return cond_mask

def interpolation_mask_batch(observed_mask):
    """
    Generates a batch of masks for interpolation task by randomly selecting timestamps to mask across all features.
    
    Parameters:
    - observed_mask (Tensor): A tensor indicating observed values.
    
    Returns:
    - Tensor: A mask tensor for interpolation tasks.
    """
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask ### array like observed mask filled with random values
    total_timestamps = observed_mask.shape[2]
    timestamps = np.arange(total_timestamps)
    for i in range(len(observed_mask)):
        mask_timestamp = np.random.choice(
            timestamps
        )
        rand_for_mask[i][:,mask_timestamp] = -1
    cond_mask = (rand_for_mask > 0).float()
    return cond_mask
    

def forecasting_mask_batch(observed_mask):
    """
    Generates a batch of masks for forecasting task by masking out all future values beyond a randomly selected start
    point in the sequence (30% timestamps at most).
    
    Parameters:
    - observed_mask (Tensor): A tensor indicating observed values.
    
    Returns:
    - Tensor: A mask tensor for forecasting tasks.
    """
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask ### array like observed mask filled with random values
    total_timestamps = observed_mask.shape[2]
    start_pred_timestamps = round(total_timestamps/3)
    timestamps = np.arange(total_timestamps)[start_pred_timestamps:]
    for i in range(len(observed_mask)):
        start_forecast_mask = np.random.choice(
            timestamps
        )
        rand_for_mask[i][:,-start_forecast_mask:] = -1
    cond_mask = (rand_for_mask > 0).float()
    return cond_mask
    

def forecasting_imputation_mask_batch(observed_mask):
    """
    Generates a batch of masks for forecasting/imputation task by masking out all 
    future values for a random subset of features beyond a randomly selected start
    point in the sequence (30% timestamps at most).
    
    Parameters:
    - observed_mask (Tensor): A tensor indicating observed values.
    
    Returns:
    - Tensor: A mask tensor for forecasting tasks.
    """
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask ### array like observed mask filled with random values
    total_timestamps = observed_mask.shape[2]
    start_pred_timestamps = round(total_timestamps/3)
    timestamps = np.arange(total_timestamps)[start_pred_timestamps:]

    for i in range(len(observed_mask)):
        batch_indices = list(np.arange(0, len(rand_for_mask[i])))
        n_keep_dims = random.choice([1, 2, 3]) # pick how many dims to keep unmasked
        keep_dims_idx = random.sample(batch_indices, n_keep_dims) # choose the dims to keep
        mask_dims_idx = [i for i in batch_indices if i not in keep_dims_idx] # choose the dims to mask
        start_forecast_mask = np.random.choice(
            timestamps
        )
        rand_for_mask[i][mask_dims_idx, -start_forecast_mask:] = -1
    cond_mask = (rand_for_mask > 0).float()
    return cond_mask


def imputation_mask_sample(observed_mask):
    """
    Generates a mask for imputation for a single sample, similar to `imputation_mask_batch` but for an individual sample.
    
    Parameters:
    - observed_mask (Tensor): A tensor indicating observed values for a single sample.
    
    Returns:
    - Tensor: A mask tensor for imputation for the sample.
    """
    ## Observed mask of shape KxL
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask ### array like observed mask filled with random values   
    
    rand_for_mask = rand_for_mask.reshape(-1)
    min_value, max_value = 0.1, 0.9
    sample_ratio = min_value + (max_value - min_value)*np.random.rand()
    num_observed = observed_mask.sum().item()
    num_masked = round(num_observed * sample_ratio)
    rand_for_mask[rand_for_mask.topk(num_masked).indices] = -1
    cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
    return cond_mask

def interpolation_mask_sample(observed_mask):
    """
    Generates a mask for interpolation for a single sample by randomly selecting a timestamp to mask.
    
    Parameters:
    - observed_mask (Tensor): A tensor indicating observed values for a single sample.
    
    Returns:
    - Tensor: A mask tensor for interpolation for the sample.
    """
    ## Observed mask of shape KxL
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask ### array like observed mask filled with random values
    total_timestamps = observed_mask.shape[1]
    timestamps = np.arange(total_timestamps)
    
    mask_timestamp = np.random.choice(
        timestamps
    )
    rand_for_mask[:,mask_timestamp] = -1
    cond_mask = (rand_for_mask > 0).float()
    return cond_mask
    

def forecasting_mask_sample(observed_mask):
    """
    Generates a mask for forecasting for a single sample by masking out all future values beyond a selected timestamp.
    
    Parameters:
    - observed_mask (Tensor): A tensor indicating observed values for a single sample.
    
    Returns:
    - Tensor: A mask tensor for forecasting for the sample.
    """
    ## Observed mask of shape KxL
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask ### array like observed mask filled with random values
    total_timestamps = observed_mask.shape[1]
    
    start_pred_timestamps = round(total_timestamps/3)
    timestamps = np.arange(total_timestamps)[-start_pred_timestamps:]
    
    start_forecast_mask = np.random.choice(
        timestamps
    )
    rand_for_mask[:,start_forecast_mask:] = -1
    cond_mask = (rand_for_mask > 0).float()
    
    return cond_mask



def get_mask_equal_p_sample(observed_mask):
    """
    IIF mix masking strategy.
    Generates masks for a batch of samples where each sample has an equal probability of being assigned one of the
    three mask types: imputation, interpolation, or forecasting.
    
    Parameters:
    - observed_mask (Tensor): A tensor indicating observed values for a batch of samples.
    
    Returns:
    - Tensor: A batch of masks with a mix of the three types.
    """
    B, K, L = observed_mask.shape
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    for i in range(B):
        
        threshold = 1/3

        imp_mask = imputation_mask_sample(observed_mask[i])
        p = np.random.rand()  # missing probability at random

        if p<threshold: 

            cond_mask = imp_mask

        elif p<2*threshold:

            cond_mask = interpolation_mask_sample(imp_mask)

        else:

            cond_mask = forecasting_mask_sample(imp_mask)

        rand_for_mask[i]=cond_mask
    
    return rand_for_mask
                            
def get_mask_probabilistic_layering(observed_mask):
    """
    Mix masking strategy.
    Generates masks for a batch of samples using a probabilistic layering approach where masks are applied in a 
    random order and with a random chance, potentially layering multiple types of masks.
    
    Parameters:
    - observed_mask (Tensor): A tensor indicating observed values for a batch of samples.
    
    Returns:
    - Tensor: A batch of masks generated by probabilistic layering of mask types.
    """

    B, K, L = observed_mask.shape
    types = ['imputation', 'forecasting','interpolation']
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    for i in range(B):
        random.shuffle(types)
        mask = observed_mask[i]
        # missing rate
        m_initial = torch.sum(torch.eq(mask, 0))/(K*L)
        m = m_initial
        for mask_type in types:
            p = np.random.rand()
            if mask_type == types[-1] and m==m_initial:
                p = 1
            if p>0.5:
                if mask_type == 'imputation':
                    mask = imputation_mask_sample(mask)
                elif mask_type == 'interpolation':
                    mask = interpolation_mask_sample(mask)
                else:
                    mask = forecasting_mask_sample(mask)
                    
                m = torch.sum(torch.eq(mask, 0))/(K*L)
    
        rand_for_mask[i]=mask
    
    return rand_for_mask

def pattern_mask_batch(observed_mask):
    """
    Generates a batch of masks based on a predetermined pattern or a random choice between imputation mask and a
    previously used mask pattern. Used for finetuning TSDE on PM25 dataset.
    
    Parameters:
    - observed_mask (Tensor): A tensor indicating observed values for a batch of samples.
    
    Returns:
    - Tensor: A batch of masks where each mask is either an imputation mask or follows a specific pattern.
    """
    pattern_mask = observed_mask
    rand_mask = imputation_mask_batch(observed_mask)

    cond_mask = observed_mask.clone()  ### Gradients can flow back to observed_mask
    for i in range(len(cond_mask)):
        mask_choice = np.random.rand()
        if mask_choice > 0.5:
            cond_mask[i] = rand_mask[i]
        else:  # draw another sample for histmask (i-1 corresponds to another sample) ###### Not randomly sampled?
            cond_mask[i] = cond_mask[i] * pattern_mask[i - 1] 
    return cond_mask


