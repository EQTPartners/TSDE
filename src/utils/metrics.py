import numpy as np
import torch
from sklearn.metrics import auc


import matplotlib.pyplot as plt


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    """
    Calculates the quantile loss for a given forecast and target values based on a specific quantile.

    Parameters:
    - target: Torch tensor containing the observed values.
    - forecast: Torch tensor containing the predicted values.
    - q: Float, representing the quantile for which the loss is calculated.
    - eval_points: Torch tensor representing the evaluation points for the calculation.

    Returns:
    - Float representing the calculated quantile loss.
    """
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    """
    Calculates the denominator used in the CRPS and CRPS-sum calculation, based on the absolute sum of the target values used for evaluation.

    Parameters:
    - target: Torch tensor containing the target values.
    - eval_points: Torch tensor representing the evaluation points for the calculation.

    Returns:
    - Torch tensor representing the denominator value.
    """
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    """
    Calculates the CRPS based on quantile loss for multiple quantiles.

    Parameters:
    - target: Torch tensor containing the target values.
    - forecast: Torch tensor containing the predicted values.
    - eval_points: Torch tensor representing the evaluation points for the calculation.
    - mean_scaler: Float, the mean value used for scaling the target and forecast back to their original values.
    - scaler: Float, the scale value used for scaling the target and forecast back to their original values.

    Returns:
    - Float representing the calculated CRPS.
    """
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):
    """
    Calculates the CRPS for the sum of the target and predicted values across all features.

    Parameters:
    - target: Torch tensor containing the target values.
    - forecast: Torch tensor containing the predicted values.
    - eval_points: Torch tensor representing the evaluation points for the calculation.
    - mean_scaler: Float, the mean value used for scaling the target and predictions back to their original values.
    - scaler: Float, the scale value used for scaling the target and predictions back to their original values.

    Returns:
    - Float representing the calculated CRPS-sum.
    """
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler
    target_sum = torch.sum(target, dim = 2).unsqueeze(2)
    forecast_sum = torch.sum(forecast, dim = 3).unsqueeze(3)
    eval_points_sum = torch.mean(eval_points, dim=2).unsqueeze(2)

    crps_sum = calc_quantile_CRPS(target_sum, forecast_sum, eval_points_sum, 0, 1)
    return crps_sum

    
def save_roc_curve(fpr, tpr, foldername):
    """
    Generates and saves an ROC curve to the specified file path.

    Parameters:
    y_true (array-like): True binary labels.
    y_scores (array-like): Scores assigned by the classifier.
    file_path (str): Path where the ROC curve image will be saved.

    Returns:
    str: The file path where the image was saved.
    """
 
    roc_auc = auc(fpr, tpr)

    # Plotting and saving the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(foldername+'roc.png')
    plt.close()