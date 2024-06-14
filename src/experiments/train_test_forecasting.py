import argparse
import torch
import json
import yaml
import os
import sys


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

from data_loader.forecasting_dataloader import get_dataloader_forecasting
from tsde.main_model import TSDE_Forecasting
from utils.utils import train, evaluate, gsutil_cp, set_seed

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description="TSDE-Forecasting")
parser.add_argument("--config", type=str, default="base_forecasting.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument('--linear', action='store_true', help='Linear mode flag')
parser.add_argument('--sample_feat', action='store_true', help='Sample feature flag')


parser.add_argument("--dataset", type=str, default='Electricity')
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--run", type=int, default=1)
parser.add_argument("--mix_masking_strategy", type=str, default='equal_p', help="Mix masking strategy (equal_p or probabilistic_layering)")

args = parser.parse_args()
print(args)

path = "src/config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["mix_masking_strategy"] = args.mix_masking_strategy

    
print(json.dumps(config, indent=4))

set_seed(args.seed)

if args.dataset == "Electricity": 
    foldername = "./save/Forecasting/" + args.dataset + "/n_samples_" + str(args.nsample) + "_run_" + str(args.run) + '_linear_' + str(args.linear) + '_sample_feat_' + str(args.sample_feat)+"/"
    model = TSDE_Forecasting(config, args.device, target_dim=370, sample_feat=args.sample_feat).to(args.device)
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader_forecasting(
    dataset_name='electricity',
    train_length=5833,
    skip_length=370*6,
    batch_size=config["train"]["batch_size"],
    device= args.device,
)

elif args.dataset == "Solar":
    foldername = "./save/Forecasting/" + args.dataset + "/n_samples_" + str(args.nsample) + "_run_" + str(args.run) + '_linear_' + str(args.linear) + '_sample_feat_' + str(args.sample_feat)+"/"
    model = TSDE_Forecasting(config, args.device, target_dim=137, sample_feat=args.sample_feat).to(args.device)
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader_forecasting(
    dataset_name='solar',
    train_length=7009,
    skip_length=137*6,
    batch_size=config["train"]["batch_size"],
    device= args.device,
)
    
elif args.dataset == "Traffic":
    foldername = "./save/Forecasting/" + args.dataset + "/n_samples_" + str(args.nsample) + "_run_" + str(args.run) + '_linear_' + str(args.linear) + '_sample_feat_' + str(args.sample_feat)+"/"
    model = TSDE_Forecasting(config, args.device, target_dim=963, sample_feat=args.sample_feat).to(args.device)
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader_forecasting(
    dataset_name='traffic',
    train_length=4001,
    skip_length=963*6,
    batch_size=config["train"]["batch_size"],
    device= args.device,
)
    
elif args.dataset == "Taxi":
    foldername = "./save/Forecasting/" + args.dataset + "/n_samples_" + str(args.nsample) + "_run_" + str(args.run) + '_linear_' + str(args.linear) + '_sample_feat_' + str(args.sample_feat)+"/"
    model = TSDE_Forecasting(config, args.device, target_dim=1214, sample_feat=args.sample_feat).to(args.device)
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader_forecasting(
    dataset_name='taxi',
    train_length=1488,
    skip_length=1214*55,
    test_length=24*56,
    history_length=48, 
    batch_size=config["train"]["batch_size"],
    device= args.device,
)
    
elif args.dataset == "Wiki":
    foldername = "./save/Forecasting/" + args.dataset + "/n_samples_" + str(args.nsample) + "_run_" + str(args.run) + '_linear_' + str(args.linear) + '_sample_feat_' + str(args.sample_feat)+"/"
    model = TSDE_Forecasting(config, args.device, target_dim=2000, sample_feat=args.sample_feat).to(args.device)
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader_forecasting(
    dataset_name='wiki',
    train_length=792,
    skip_length=9535*4,
    test_length=30*5,
    valid_length=30*5,
    history_length=90, 
    pred_length=30,
    batch_size=config["train"]["batch_size"],
    device= args.device,
)

else: 
    print()


print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)


with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)


if args.modelfolder == "":
    loss_path = foldername + "/losses.txt"
    with open(loss_path, "a") as file:
        file.write("Pretraining"+"\n")
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        foldername=foldername,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        eval_epoch_interval=200,
    )
    if config["finetuning"]["epochs"]!=0:
        print("Finetuning")
        ## Fine Tuning   
        with open(loss_path, "a") as file:
            file.write("Finetuning"+"\n")
        checkpoint_path = foldername + "model.pth"
        model.load_state_dict(torch.load(checkpoint_path))
        config["train"]["epochs"]=config["finetuning"]["epochs"]
        train(
           model,
           config["train"],
           train_loader,
           valid_loader=valid_loader,
           foldername=foldername,
           mode = 'Forecasting',
        )

else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth", map_location=args.device))

evaluate(model, test_loader, nsample=args.nsample, foldername=foldername, scaler=scaler, mean_scaler=mean_scaler,save_samples = True)

