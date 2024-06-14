import argparse
import torch
import json
import yaml
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

# Import necessary modules from your project
from data_loader.elec_tslib_dataloader import get_dataloader_elec
from tsde.main_model import TSDE_Forecasting
from utils.utils import train, evaluate, gsutil_cp, set_seed

def main():
    # Command line arguments for configuration, could be expanded as needed
    parser = argparse.ArgumentParser(description="Run forecasting model")
    parser.add_argument('--config', type=str, default='base_forecasting.yaml', help='Path to configuration yaml')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to run the model on')
    parser.add_argument('--nsample', type=int, default=100, help='Number of samples')
    parser.add_argument('--hist_length', type=int, default=96, help='History window length')
    parser.add_argument('--pred_length', type=int, default=192, help='Prediction window length')
    parser.add_argument('--run', type=int, default=100200000, help='Run identifier')
    parser.add_argument('--linear', action='store_true', help='Linear mode flag')
    parser.add_argument('--sample_feat', action='store_true', help='Sample feature flag')
    parser.add_argument('--seed', type=int, default=1, help='Seed for random number generation')
    parser.add_argument('--load', type=str, default=None, help='Path to pretrained model')
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load config
    path = "src/config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Setup model folder path
    foldername = f"./save/Forecasting/TSLIB_Elec/n_samples_{args.nsample}_run_{args.run}_linear_{args.linear}_sample_feat_{args.sample_feat}/"
    os.makedirs(foldername, exist_ok=True)

    # Save configuration to the model folder
    with open(os.path.join(foldername, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Model setup
    model = TSDE_Forecasting(config, args.device, target_dim=321, sample_feat=args.sample_feat).to(args.device)
    train_loader, valid_loader, test_loader, _ = get_dataloader_elec(
        pred_length=args.pred_length, 
        history_length=args.hist_length,
        batch_size=config["train"]["batch_size"],
        device=args.device,
    )
    if args.load is None:
        # Start training
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            foldername=foldername,
            nsample=args.nsample,
            scaler=1,
            mean_scaler=0,
            eval_epoch_interval=20000,
        )
    else:
        model.load_state_dict(torch.load("./save/" + args.load + "/model.pth", map_location=args.device))

    evaluate(model, test_loader, nsample=args.nsample, foldername=foldername)
if __name__ == "__main__":
    main()
