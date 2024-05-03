import argparse
import torch
import json
import yaml
import os
import sys
import torch.nn as nn

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

from data_loader.physio_dataloader import get_dataloader_physio, get_physio_dataloader_for_classification
from tsde.main_model import TSDE_Physio
from utils.utils import train, evaluate, gsutil_cp, set_seed, finetune, evaluate_finetuning

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description="TSDE-Imputation-Classification")
parser.add_argument("--config", type=str, default="base_classification.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)

parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--run", type=int, default=1)
parser.add_argument("--mix_masking_strategy", type=str, default='equal_p', help="Mix masking strategy (equal p or probabilistic layering)")
parser.add_argument("--disable_finetune", action="store_true")

## Args for physio
parser.add_argument("--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])")
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument("--physionet_classification", type=bool, default=True)

args = parser.parse_args()
print(args)

path = "src/config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["mix_masking_strategy"] = args.mix_masking_strategy
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

set_seed(args.seed)


foldername = "./save/Imputation-Classification/" + 'PhysioNet' + "/n_samples_" + str(args.nsample) + "_run_" + str(args.run) + '_missing_ratio_' + str(args.testmissingratio) +"/"
model = TSDE_Physio(config, args.device).to(args.device)
train_loader, valid_loader, test_loader = get_dataloader_physio(seed=args.seed, nfold=args.nfold, batch_size=config["train"]["batch_size"], missing_ratio=config["model"]["test_missing_ratio"])
scaler = 1
mean_scaler = 0
mode = 'Imputation'

    
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)

with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

if args.modelfolder == "":
    os.makedirs(foldername+'Pretrained/', exist_ok=True)
    loss_path = foldername + "losses.txt"
    with open(loss_path, "a") as file:
        file.write("Pretraining"+"\n")
    ## Pre-training
    train(model, config["train"], train_loader, valid_loader=valid_loader, test_loader=test_loader, foldername=foldername+'Pretrained/', nsample=args.nsample,
       scaler=scaler, mean_scaler=mean_scaler, eval_epoch_interval=100000,physionet_classification=args.physionet_classification)
     
    ## Save imputed time series in Train, Validation and Test sets    
    evaluate(
    model,
    train_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername+'Pretrained/',
    save_samples = True,
    physionet_classification=True,
    set_type = 'Train'
    )

    evaluate(
        model,
        valid_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername+'Pretrained/',
        save_samples = True,
        physionet_classification=True,
        set_type = 'Val'
    )
    evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername+'Pretrained/',
        save_samples = True,
        physionet_classification=True,
        set_type = 'Test'
    )


    ## Prepare dataloaders for classification head finetuning
    train_loader_classification, valid_loader_classification, test_loader_classification = get_physio_dataloader_for_classification(filename=foldername+'Pretrained/', batch_size=config["train"]["batch_size"])
    model.load_state_dict(torch.load(foldername + "Pretrained/model.pth", map_location=args.device))

    
else:
    # Load pretrained and dataloaders of imputed MTS
    train_loader_classification, valid_loader_classification, test_loader_classification = get_physio_dataloader_for_classification(filename=args.modelfolder+"Pretrained/", batch_size=config["train"]["batch_size"])
    model.load_state_dict(torch.load(args.modelfolder + "Pretrained/model.pth", map_location=args.device))
    
print(args.disable_finetune)
if not args.disable_finetune:
    for name, param in model.named_parameters():
            # Freeze all parameters
            param.requires_grad = False


    for name, param in model.mlp.named_parameters():
        param.requires_grad = True
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")

    ## Finetune the classifier head    
    finetune(model, config["finetuning"], train_loader_classification, criterion = nn.CrossEntropyLoss(), foldername=foldername) 
else:
    model.load_state_dict(torch.load(args.modelfolder + "model.pth", map_location=args.device))


## Evaluate the classification
evaluate_finetuning(model, train_loader_classification, test_loader_classification, foldername=foldername)


