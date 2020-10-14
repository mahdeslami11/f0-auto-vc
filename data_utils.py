import random
from math import ceil
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import os
import glob
from hparams import hparams
import pickle

from utils import quantize_f0_numpy


class VCDataset(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, data_dir, hparams, is_train=True):
        self.spk_path = [p for p in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(p)]
        self.is_train = is_train
        self.npz_path, self.metadata = self.get_npz_path(self.spk_path)
        self.n_speakers = hparams.n_speakers

        assert len(self.npz_path) > 0, "npz 파일 탐색 실패"
        random.seed(1234)
        random.shuffle(self.npz_path)

    def get_npz_path(self, spk_path):
        metadata = {}
        npz_path = []
        for spk in spk_path:
            if self.is_train:
                spk_npz = glob.glob(os.path.join(spk, "train", "*.npz"))
            else:
                spk_npz = glob.glob(os.path.join(spk, "test", "*.npz"))
            npz_path += spk_npz

        return npz_path, metadata

    def get_sample(self, npz_path):
        # separate filename and text
        npz = np.load(npz_path)
        mel = npz['mel'].T if npz['mel'].shape[0] == 80 else npz['mel']
        f0 = npz['f0']
        spk_label = self.get_speaker(npz['spk_label'].item())

        return (mel, f0, spk_label)

    def get_speaker(self, speaker):
        speaker_vector = np.zeros(self.n_speakers)
        speaker_vector[int(speaker)] = 1
        return speaker_vector.astype(dtype=np.float32)


    def __getitem__(self, index):
        return self.get_sample(self.npz_path[index])

    def __len__(self):
        return len(self.npz_path)

class AutoVCCollate():
    def __init__(self, hparams):
        self.seq_len = hparams.seq_len

    def __call__(self, batch):
        # batch : B * (mel, f0, spk)
        mels = [b[0] for b in batch]
        f0s = [b[1] for b in batch]
        spk = [b[2] for b in batch]


        mel_seg = []
        f0_seg = []
        speaker_embeddings = []
        for mel, f0, spk_emb in zip(mels, f0s, spk):
            frame_len = mel.shape[0]
            if frame_len < self.seq_len:
                len_pad = self.seq_len - frame_len
                x = np.pad(mel, ((0, len_pad), (0, 0)), 'constant')
                p = np.pad(f0, ((0, len_pad)), 'constant', constant_values=-1e10)
            else:
                start = np.random.randint(frame_len - self.seq_len + 1)
                x = mel[start:start + self.seq_len]
                p = f0[start:start + self.seq_len]

            quantized_p, _ = quantize_f0_numpy(p, num_bins=hparams.pitch_bin)

            mel_seg.append(x)
            f0_seg.append(quantized_p)
            speaker_embeddings.append(spk_emb)


        out = {"mel": torch.FloatTensor(mel_seg),
               "quantized_p": torch.FloatTensor(f0_seg),
               "spk": torch.FloatTensor(speaker_embeddings),
               }

        return out

class VCTestSet:
    def __init__(self, data_dir, hparams):
        self.n_speakers = hparams.n_speakers
        self.spk_path = glob.glob(os.path.join(data_dir, '*'))
        assert len(self.spk_path) > 0, "speaker 탐색 실패"

    def get_random_pair(self):
        random.seed(1234)
        random.shuffle(self.spk_path)
        src_spk_path = self.spk_path[0]
        trg_spk_path = self.spk_path[1]

        src_spk_name = os.path.basename(src_spk_path)
        trg_spk_name = os.path.basename(trg_spk_path)

        src_npz_path = self.get_first_npz(src_spk_path)
        trg_npz_path = self.get_first_npz(trg_spk_path)

        return (src_spk_name, src_npz_path, trg_spk_name, trg_npz_path)

    def get_random_npz(self, spk_path):
        npz_path = glob.glob(os.path.join(spk_path, "test/*.npz"))
        idx = np.random.randint(0, len(npz_path))

        return npz_path[idx]

    def get_first_npz(self, spk_path):
        npz_path = glob.glob(os.path.join(spk_path, "test/*.npz"))

        return npz_path[0]

    def parse_npz(self, npz_path):
        npz = np.load(npz_path)
        mel = torch.from_numpy(npz['mel'])
        f0 = torch.from_numpy(npz['f0'])
        spk = torch.from_numpy(self.get_speaker(npz['speaker'].item()))

        mel = mel.unsqueeze(0).float()
        f0 = f0.unsqueeze(0).float()
        spk = spk.unsqueeze(0).float()

        return (mel, f0, spk)

    def get_speaker(self, speaker):
        speaker_vector = np.zeros(self.n_speakers)
        speaker_vector[int(speaker)] = 1
        return speaker_vector.astype(dtype=np.float32)


def prepare_dataloaders(data_path, hparams):
    # Get data, data loaders and collate function ready
    trainset = VCDataset(data_path, hparams, is_train=True)
    valset = VCDataset(data_path, hparams, is_train=False)
    collate_fn = AutoVCCollate(hparams)

    train_loader = DataLoader(trainset, num_workers=1, shuffle=True,
                              batch_size=hparams.batch_size,
                              drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(valset, num_workers=1, shuffle=False,
                              batch_size=hparams.val_batch_size,
                              drop_last=True, collate_fn=collate_fn)
    test_set = VCTestSet(data_path, hparams)

    return train_loader, val_loader, test_set

if __name__ == "__main__":
    train_loader, val_loader, test_set = prepare_dataloaders("/hd0/f0-autovc/preprocessed/sr16000_npz", hparams)
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    out = train_iter.__next__()
    """
    out = {"mel": torch.FloatTensor(mel_targets),
               "phoneme": torch.FloatTensor(phonemes),
               "D": torch.FloatTensor(Ds),
               "mel_pos": torch.LongTensor(mel_pos),
               "mel_max_len": max_mel_len,
               "D_max_len": max_D_len
               }
    """

    print(out["mel"].size())
    print(out["quantized_p"].size())
    print(out["spk"].size())
