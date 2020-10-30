import tensorflow as tf
import numpy as np
import math
import torch

# NOTE: If you want full control for model architecture. please take a look
# at the code and change whatever you want. Some hyper parameters are hardcoded.

# Default hyperparameters:
hparams = tf.contrib.training.HParams(
    exp='AutoVC',
    n_speakers=12,

    device= 'cuda' if torch.cuda.is_available() else 'cpu',
    #device= 'cpu',

    # preprocessing 때 사용
    used_spks=None,

    ########## Audio ######################################
    sample_rate=16000,  #

    # shift can be specified by either hop_size(?곗꽑) or frame_shift_ms
    hop_size=256,  # frame_shift_ms = 12.5ms
    fft_size=1024,
    win_size=1024,  # 50ms
    num_mels=80,

    min_level_db=-100,
    ref_level_db=16,

    rescaling=True,
    rescaling_max=0.999,

    trim_silence=False,  # Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
    # M-AILABS (and other datasets) trim params (there parameters are usually correct for any data, but definitely must be tuned for specific speakers)
    trim_fft_size=1024,
    trim_hop_size=256,
    trim_top_db=20,

    # filter parameter
    cutoff=30,
    order=5,

    # mel-basis parameters
    fmin=90,
    fmax=7600,

    ########## Model Parameters ######################################
    # input
    seq_len = 128,

    # Model
    dim_neck = 32,
    dim_emb = 12,
    dim_pre = 512,
    freq = 16,
    pitch_bin = 256,

    ################################################################################
    # Training:
    batch_size = 2, # it is equal to N
    val_batch_size = 2,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-8,
    amsgrad=False,
    initial_learning_rate= 1e-3,
    final_learning_rate = 1e-6,
    n_warmup_steps=4000, # ScheduledOptim : 4000, Exponential : 40000
    decay_rate = 0.000005,
    decay_step = 1000000,
    nepochs=500,
    ################################################################################
    # Save
    # per-epoch interval
    save_every=20,
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
