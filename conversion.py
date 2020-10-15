import os
import pickle
import torch
import numpy as np
from utils import pad_seq, make_onehot, quantize_f0_numpy
from hparams import hparams
import argparse
import importlib
import glob
import itertools
import torch
from tqdm import tqdm

def conversion(args, net, device='cuda'):
    assert os.path.isdir(args.data_dir), 'Cannot found data dir : {}'.format(args.data_dir)

    all_spk_path = [p for p in glob.glob(os.path.join(args.data_dir, '*')) if os.path.isdir(p)]
    all_test_samples = [glob.glob(os.path.join(p, 'test', '*.npz'))[0] for p in all_spk_path]
    os.makedirs(args.out_dir, exist_ok=True)

    all_pair = itertools.product(all_test_samples, all_test_samples)
    for src, trg in tqdm(all_pair, desc="converting voices"):
        src_name = src.split('/')[-3]
        trg_name = trg.split('/')[-3]
        src_npz = np.load(src)
        trg_npz = np.load(trg)

        x = src_npz['mel']
        p = src_npz['f0'][:, np.newaxis]
        emb_src_np = make_onehot(src_npz['spk_label'].item(), hparams.n_speakers)
        emb_trg_np = make_onehot(trg_npz['spk_label'].item(), hparams.n_speakers)

        x_padded, pad_len = pad_seq(x, base=hparams.freq, constant_values=None)
        p_padded, pad_len = pad_seq(p, base=hparams.freq, constant_values=-1e10)

        quantized_p, _ = quantize_f0_numpy(p_padded[:, 0], num_bins=hparams.pitch_bin)

        x_src = torch.from_numpy(x_padded).unsqueeze(0).to(device)
        p_src = torch.from_numpy(quantized_p).unsqueeze(0).to(device)
        emb_src = torch.from_numpy(emb_src_np).unsqueeze(0).to(device)
        emb_trg = torch.from_numpy(emb_trg_np).unsqueeze(0).to(device)

        if args.model == 'autovc':
            out, out_psnt, _ = net(x_src, emb_src, emb_trg)
        elif args.model == 'autovc-f0':
            out, out_psnt, _ = net(x_src, p_src, emb_src, emb_trg)
        else:
            print("Wrong model name : {}".format(args.model))

        print(out_psnt)

        if pad_len == 0:
            out_mel = out_psnt.squeeze().detach().cpu().numpy()[:, :]
        else:
            out_mel = out_psnt.squeeze().detach().cpu().numpy()[:-pad_len, :]
        src_mel = src_npz['mel']
        trg_mel = trg_npz['mel']

        np.save(os.path.join(args.out_dir, '{}-{}-feats.npy'.format(src_name, os.path.splitext(src.split('/')[-1])[0])), src_mel)
        np.save(os.path.join(args.out_dir, '{}-{}-feats.npy'.format(trg_name, os.path.splitext(trg.split('/')[-1])[0])), trg_mel)
        np.save(os.path.join(args.out_dir, '{}-to-{}-{}.npy'.format(src_name, trg_name, os.path.splitext(src.split('/')[-1])[0])), out_mel)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='autovc-f0', help="set 'autovc' or 'autovc-f0'")
    parser.add_argument('--data_dir', type=str, default='/hd0/f0-autovc/preprocessed/sr16000_npz/')
    parser.add_argument('--out_dir', type=str, default='generated')
    parser.add_argument('--checkpoint', type=str, default='/hd0/f0-autovc/exp/autovc_f0_voicesplit_freq8/model/60.pkl')
    parser.add_argument('--architecture', type=str, default='architectures/arch_autovc_f0.py')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # architecture
    arch_name = os.path.splitext(args.architecture)[0].replace("/", ".")
    arch = importlib.import_module(arch_name)
    print(" [*] Load architecture : {}".format(arch_name))

    # load model
    net = arch.get_network(hparams)
    net = net['net'].to(device)

    assert os.path.isfile(args.checkpoint), 'Cannot found model checkpoint : {}'.format(args.checkpoint)
    dd = torch.load(args.checkpoint)
    net.load_state_dict(dd['net'], strict=True)
    net.eval()

    conversion(args, net, device)