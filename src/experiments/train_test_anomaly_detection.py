import argparse
import torch
import json
import yaml
import os
import sys
import random
import numpy as np
import torch.nn as nn

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

from data_loader.anomaly_detection_dataloader import anomaly_detection_dataloader
from tsde.main_model import TSDE_AD
from utils.utils import train, finetune, evaluate_finetuning, gsutil_cp, set_seed

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description="TSDE-Anomaly Detection")
parser.add_argument("--config", type=str, default="base_ad.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--disable_finetune", action="store_true")


parser.add_argument("--dataset", type=str, default='SMAP')
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--run", type=int, default=1)
parser.add_argument("--mix_masking_strategy", type=str, default='equal_p', help="Mix masking strategy (equal_p or probabilistic_layering)")
parser.add_argument("--anomaly_ratio", type=float, default=1, help="Anomaly ratio")
args = parser.parse_args()
print(args)

path = "src/config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["mix_masking_strategy"] = args.mix_masking_strategy

print(json.dumps(config, indent=4))

set_seed(args.seed)


foldername = "./save/Anomaly_Detection/" + args.dataset + "/run_" + str(args.run) +"/"
model = TSDE_AD(target_dim = config["embedding"]["num_feat"], config = config, device = args.device).to(args.device)
train_loader, valid_loader, test_loader = anomaly_detection_dataloader(dataset_name = args.dataset, batch_size = config["train"]["batch_size"])
anomaly_ratio = args.anomaly_ratio    
    
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)

with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

if args.modelfolder == "":
    loss_path = foldername + "/losses.txt"
    with open(loss_path, "a") as file:
        file.write("Pretraining"+"\n")
    ## Pre-training
    train(model, config["train"], train_loader, foldername=foldername, normalize_for_ad=True)
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth", map_location=args.device))


if not args.disable_finetune:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.conv.parameters():
        param.requires_grad = True

    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")

    finetune(model, config["finetuning"], train_loader, criterion = nn.MSELoss(), foldername=foldername, task='anomaly_detection', normalize_for_ad=True) 
evaluate_finetuning(model, train_loader, test_loader, anomaly_ratio = anomaly_ratio, foldername=foldername, task='anomaly_detection', normalize_for_ad=True)
