import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torch import optim
from itertools import chain
from solver.base import Base
import argparse
from hparams import hparams
from utils import prepare_dirs
from data_utils import prepare_dataloaders
from plotting_utils import plot_spectrogram_to_numpy, plot_f0_to_numpy
import importlib
import os

class AutoVC(Base):
    def __init__(self,
                 architecture,
                 model_dir,
                 log_dir,
                 sample_dir):
        super(AutoVC, self).__init__(model_dir, log_dir, sample_dir)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.device = 'cpu'

        # load architecture
        arch = importlib.import_module(architecture)

        nets = arch.get_network(hparams)
        self.net = nets['net'].to(self.device)

        opt_args = {'lr': hparams.initial_learning_rate, 'betas': (hparams.adam_beta1, hparams.adam_beta2)}
        g_params = self.net.parameters()
        optim_g = optim.Adam(filter(lambda p: p.requires_grad, g_params), **opt_args)

        self.optim = {
            'g': optim_g
        }

        self.last_epoch = 0
        self.load_strict = True

    def _get_stats(self, dict_, mode):
        stats = OrderedDict({})
        for key in dict_.keys():
            stats[key] = np.mean(dict_[key])
        return stats

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optim['g'].zero_grad()

    def _train(self):
        self.net.train()

    def _eval(self):
        self.net.eval()

    def train_on_instance(self,
                          x_real,
                          p_real,
                          emb_org,
                          **kwargs):
        self._train()
        # Identity mapping loss
        x_identic, x_identic_psnt, code_real = self.net(x_real, p_real, emb_org, emb_org)
        g_loss_id = F.mse_loss(x_real, x_identic)
        g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt)

        # Code semantic loss.
        code_reconst = self.net(x_identic_psnt, None, emb_org, None)
        g_loss_cd = F.l1_loss(code_real, code_reconst)

        # Backward and optimize.
        g_loss = g_loss_id + g_loss_id_psnt + g_loss_cd
        self.reset_grad()
        g_loss.backward()
        self.optim['g'].step()

        ## ----------------------------------------------
        ## Collecting losses and outputs
        ## ----------------------------------------------
        losses = {
            'G/loss_id': g_loss_id.item(),
            'G/loss_id_psnt': g_loss_id_psnt.item(),
            'G/loss_cd': g_loss_cd.item()
        }

        outputs = {
            'recon': x_identic_psnt.detach().cpu(),
        }

        return losses, outputs

    def eval_on_instance(self,
                         x_real,
                         p_real,
                         emb_org,
                         **kwargs):
        self._eval()
        with torch.no_grad():
            # Identity mapping loss
            x_identic, x_identic_psnt, code_real = self.net(x_real, p_real, emb_org, emb_org)
            g_loss_id = F.mse_loss(x_real, x_identic)
            g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt)

            # Code semantic loss.
            code_reconst = self.net(x_identic_psnt, None, emb_org, None)
            g_loss_cd = F.l1_loss(code_real, code_reconst)

            # Backward and optimize.
            g_loss = g_loss_id + g_loss_id_psnt + g_loss_cd

            ## ----------------------------------------------
            ## Collecting losses and outputs
            ## ----------------------------------------------
            losses = {
                'G/loss_id': g_loss_id.item(),
                'G/loss_id_psnt': g_loss_id_psnt.item(),
                'G/loss_cd': g_loss_cd.item()
            }

            outputs = {
                'recon': x_identic_psnt.detach().cpu(),
            }

        return losses, outputs

    def prepare_batch(self, batch):
        if len(batch) != 3:
            raise Exception("Expected batch to eight element: " +
                            "mel, quantized_p, spk")

        x = batch["mel"].to(self.device)
        quantized_p = batch["quantized_p"].to(self.device)
        spk = batch["spk"].to(self.device)

        return [x, quantized_p, spk]

    def save(self, filename, epoch):
        dd = {}
        # Save the models.
        dd['net'] = self.net.state_dict()
        # Save the models' optim state.
        for key in self.optim:
            dd['optim_%s' % key] = self.optim[key].state_dict()
        dd['epoch'] = epoch
        dd['global_epoch'] = self.global_epoch
        dd['global_step'] = self.global_step
        torch.save(dd, filename)

    def load(self, filename):
        # if not self.use_cuda:
        #     map_location = lambda storage, loc: storage
        # else:
        #     map_location = None
        dd = torch.load(filename)
                        #map_location=map_location)
        # Load the models.
        self.net.load_state_dict(dd['net'], strict=self.load_strict)

        # Load the models' optim state.
        for key in self.optim:
            self.optim[key].load_state_dict(dd['optim_%s' % key])
        self.last_epoch = dd['epoch']
        self.global_epoch = dd['global_epoch']
        self.global_step = dd['global_step']

    def summary(self, outputs, epoch):
        self.logger.image_summary("reconstruction",
                                  plot_spectrogram_to_numpy("reconstruction", outputs['recon'][0].numpy().T),
                                  epoch)