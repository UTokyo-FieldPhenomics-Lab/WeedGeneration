import torch
import argparse
from engine.tester import tester

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='segan')
parser.add_argument('--model_weight_path', default='./pre_trained/segan_spsd.tar')
parser.add_argument('--dataset', default='spsd')
parser.add_argument('--n_classes', default=9)
parser.add_argument('--output_dir', default='./Images_Gen')
parser.add_argument('--unit_number', default=10)
parser.add_argument('--bottom_width', default=4)
parser.add_argument('--gf_dim', default=1024)
parser.add_argument('--z_dim', default=128)

args = parser.parse_args()
if args.model == "segan":
    from models import segan as model
elif args.model == "sngan":
    from models import segan as model
elif args.model == "wacgan_info":
    from models import wacgan_info as model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tester = tester(args, model, device)
    tester.generation()
    tester.save()