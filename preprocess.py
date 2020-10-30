import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
import importlib
from hparams import hparams, hparams_debug_string
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def preprocess(mod, in_dir, out_dir, spk_emb, spk2gen, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    mod.build_from_path(hparams, in_dir, out_dir, spk_emb, spk2gen, num_workers=num_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='voicesplit')
    parser.add_argument('--in_dir', type=str, default='/hd0/dataset/voicesplit/')
    parser.add_argument('--out_dir', type=str, default='/hd0/f0-autovc/preprocessed/sr16000_npz')
    parser.add_argument('--num_workers', type=str, default=None)
    parser.add_argument('--spk_emb', type=str, default=None, help='speaker embedding path (default: onehot)')
    parser.add_argument('--spk2gen', type=str, default="assets/spk2gen_voicesplit.pkl", help="pickle file path for converting speaker to gender")
    args = parser.parse_args()
    print(hparams_debug_string())

    if not os.path.exists(args.out_dir):
        try:
            os.mkdir(args.out_dir)
        except FileExistsError:
            print(args.out_dir, "exists")

    if not args.num_workers:
        args.num_workers = cpu_count()

    assert args.name in ["VCTK", "NIKL", "voicesplit"]
    mod = importlib.import_module('datasets.{}'.format(args.name))

    print("---------------------------------- Preprecessing starts! ----------------------------------")
    print("dataset: {}".format(args.name))
    print("load directory: {}".format(args.in_dir))
    print("output directory: {}".format(args.out_dir))

    preprocess(mod, args.in_dir, args.out_dir, args.spk_emb, args.spk2gen, args.num_workers)
    print("---------------------------------- Preprecessing is done! ----------------------------------")