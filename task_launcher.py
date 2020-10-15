import argparse
from hparams import hparams
from utils import prepare_dirs
import os
from data_utils import prepare_dataloaders
import importlib

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--name', type=str, default="autovc_voicesplit_freq16_seqlen128_l1_nof0")
    parser.add_argument('--save_path', type=str, default="/hd0/f0-autovc/exp")
    parser.add_argument('--data_dir', type=str, default='/hd0/f0-autovc/preprocessed/sr16000_npz')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--architecture', type=str, default='architectures/arch_autovc.py')
    parser.add_argument('--solver', type=str, default='solver/autovc.py')
    args = parser.parse_args()
    return args


args = parse_args()

# prepare directory
model_dir, log_dir, sample_dir = prepare_dirs(args)

# data loaders
train_loader, val_loader, testset = prepare_dataloaders(args.data_dir, hparams)

# architecture
arch = os.path.splitext(args.architecture)[0].replace("/", ".")
print(" [*] Load architecture : {}".format(arch))

solver_mod = importlib.import_module(os.path.splitext(args.solver)[0].replace("/", "."))
print(" [*] Load solver : {}".format(solver_mod))

# solver
solver = solver_mod.AutoVC(arch,
                model_dir,
                log_dir,
                sample_dir)

if args.checkpoint:
    solver.load(args.checkpoint)

solver.train(
    train_loader,
    val_loader,
    testset,
    hparams.nepochs,
    hparams.save_every,
    verbose=True)