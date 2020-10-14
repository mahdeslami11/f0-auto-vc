import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import pickle
import pkbar
import glob
import torch

import random
import soundfile as sf
from scipy import signal
from librosa.filters import mel
from numpy.random import RandomState
from pysptk import sptk
import librosa
from utils import butter_highpass
from utils import speaker_normalization
from utils import pySTFT
from hparams import hparams

mel_basis = mel(hparams.sample_rate, hparams.fft_size, fmin=hparams.fmin, fmax=hparams.fmax, n_mels=hparams.num_mels).T
min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
b, a = butter_highpass(hparams.cutoff, hparams.sample_rate, order=hparams.order)

def build_from_path(hparams, in_dir, out_dir, spk_emb_path, spk2gen_path, num_workers=16):

    executor = ProcessPoolExecutor(max_workers=num_workers)

    # load spk paths
    if hparams.used_spks is not None:
        spk_paths = [p for p in glob.glob(os.path.join(in_dir, "*")) if os.path.isdir(p) and os.path.basename(p) in hparams.used_spks]
    else:
        spk_paths = [p for p in glob.glob(os.path.join(in_dir, "*")) if os.path.isdir(p)]

    # load speaker embedding
    if spk_emb_path:
        spk_embs = pickle.load(open(spk_emb_path, 'rb'))

    # load speaker to gender
    if spk2gen_path is not None:
        spk2gen = pickle.load(open(spk2gen_path, "rb"))
    else:
        raise ValueError

    os.makedirs(out_dir, exist_ok=True)

    # preprocessing per speaker
    for i, spk_path in enumerate(spk_paths):
        spk_name = os.path.basename(spk_path)

        if spk_emb_path:
            emb_idx = -1
            for i in range(len(spk_embs)):
                if spk_embs[i][0] == spk_name:
                    emb_idx = i
                    break

        gender = spk2gen[spk_name]
        assert gender == 'M' or gender == 'F'

        # make speaker directory
        os.makedirs(os.path.join(out_dir, spk_name), exist_ok=True)
        os.makedirs(os.path.join(out_dir, spk_name, 'train'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, spk_name, 'val'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, spk_name, 'test'), exist_ok=True)

        # glob all samples for a speaker
        all_wav_path = glob.glob(os.path.join(spk_path, "*.wav"))
        random.shuffle(all_wav_path)

        total_num = len(all_wav_path)
        train_num = int(total_num * 0.95)
        val_num = total_num - train_num - 1
        test_num = 1

        pbar = pkbar.Pbar(name='loading and processing dataset', target=len(all_wav_path))

        futures = []
        for j, wav_path in enumerate(all_wav_path):
            wav_name = os.path.basename(wav_path)
            spk_emb = spk_embs[emb_idx][1] if spk_emb_path else None

            if j < train_num:
                npz_name = os.path.join(out_dir, spk_name, 'train', wav_name[:-4] + ".npz")
            elif j >= train_num and j < train_num + val_num:
                npz_name = os.path.join(out_dir, spk_name, 'val', wav_name[:-4] + ".npz")
            else:
                npz_name = os.path.join(out_dir, spk_name, 'test', wav_name[:-4] + ".npz")

            futures.append(executor.submit(partial(_processing_data, hparams, wav_path, i, spk_emb, gender, npz_name, pbar, i)))

        results = [future.result() for future in futures if future.result() is not None]

    print('Finish Preprocessing')


def _processing_data(hparams, full_path, spk_label, spk_emb, gender, npz_name, pbar, i):
    if gender == 'M':
        lo, hi = 50, 250
    elif gender == 'F':
        lo, hi = 100, 600
    else:
        raise ValueError

    prng = RandomState(int(random.random()))
    x, fs = librosa.load(full_path, sr=hparams.sample_rate)
    assert fs == hparams.sample_rate
    if x.shape[0] % hparams.hop_size == 0:
        x = np.concatenate((x, np.array([1e-06])), axis=0)
    y = signal.filtfilt(b, a, x)
    wav = y * 0.96 + (prng.rand(y.shape[0]) - 0.5) * 1e-06

    # compute spectrogram
    D = pySTFT(wav).T
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - hparams.ref_level_db
    S = (D_db + 100) / 100

    # extract f0
    f0_rapt = sptk.rapt(wav.astype(np.float32) * 32768, fs, hparams.hop_size, min=lo, max=hi, otype=2)
    index_nonzero = (f0_rapt != -1e10)
    mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
    f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)

    assert len(S) == len(f0_rapt)

    data = {
        'mel': S.astype(np.float32),
        'f0': f0_norm.astype(np.float32),
        'spk_label': spk_label
    }

    np.savez(npz_name, **data)
    pbar.update(i)

