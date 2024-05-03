import argparse
import torch
import json
import yaml
import os
import sys


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)


from data_loader.physio_dataloader import get_dataloader_physio
from tsde.main_model import TSDE_Physio
from utils.utils import train, evaluate, gsutil_cp, set_seed

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description="TSDE-Interpolation")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)


parser.add_argument("--dataset", type=str, default='PhysioNet')
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--run", type=int, default=1)
parser.add_argument("--mix_masking_strategy", type=str, default='equal_p', help="Mix masking strategy (equal_p or probabilistic_layering)")


## Args for physio
parser.add_argument("--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])")
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument("--physionet_classification", type=bool, default=False)

args = parser.parse_args()
print(args)

path = "src/config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["mix_masking_strategy"] = args.mix_masking_strategy
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

set_seed(args.seed)

if args.dataset == "PhysioNet":
    foldername = "./save/Interpolation/" + args.dataset + "/n_samples_" + str(args.nsample) + "_run_" + str(args.run) + '_missing_ratio_' + str(args.testmissingratio) +"/"
    model = TSDE_Physio(config, args.device).to(args.device)
    train_loader, valid_loader, test_loader = get_dataloader_physio(seed=args.seed, nfold=args.nfold, batch_size=config["train"]["batch_size"], missing_ratio=config["model"]["test_missing_ratio"], mode='interpolation')
    scaler = 1
    mean_scaler = 0
    mode = 'Interpolation'
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
    ## Pre-training
    train(model, config["train"], train_loader, valid_loader=valid_loader, test_loader=test_loader, foldername=foldername, nsample=args.nsample,
       scaler=scaler, mean_scaler=mean_scaler, eval_epoch_interval=500,physionet_classification=args.physionet_classification)
    if config["finetuning"]["epochs"]!=0:
        print('Finetuning')
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
           mode = mode,
        )

else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth", map_location=args.device))

evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
    save_samples = True,
    physionet_classification=args.physionet_classification

)

