import os
import time
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from plotting_utils import plot_spectrogram_to_numpy

class Base:
    def __init__(self,
                 model_dir,
                 log_dir,
                 sample_dir):
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.sample_dir = sample_dir

        self.global_epoch = 0
        self.global_step = 0

        self.build_tensorboard(self.log_dir)

    def train(self,
              itr_train,
              itr_valid,
              testset,
              epochs,
              save_every=1,
              verbose=True):

        f_mode = 'w' if not os.path.exists("%s/results.txt" % self.log_dir) else 'a'
        f = None
        if self.log_dir is not None:
            f = open("%s/results.txt" % self.log_dir, f_mode)

        try:
            for epoch in range(self.global_epoch, epochs):
                epoch_start_time = time.time()
                # Training.
                if verbose:
                    pbar = tqdm(total=len(itr_train))
                train_dict = OrderedDict({'epoch': epoch+1})
                # item, pose, id
                for b, batch in enumerate(itr_train):
                    self.global_step += 1
                    batch = self.prepare_batch(batch)
                    losses, outputs = self.train_on_instance(*batch,
                                                             iter=b+1,
                                                             global_step = self.global_step)
                    for key in losses:
                        this_key = 'train/%s' % key
                        if this_key not in train_dict:
                            train_dict[this_key] = []
                        train_dict[this_key].append(losses[key])
                        self.logger.scalar_summary(this_key, losses[key], self.global_step)
                    if verbose:
                        pbar.update(1)
                        pbar.set_postfix(self._get_stats(train_dict, 'train'))
                if verbose:
                    pbar.close()
                valid_dict = {}
                # TODO: enable valid
                if verbose:
                    pbar = tqdm(total=len(itr_valid))
                # Validation.
                valid_dict = OrderedDict({})
                for b, valid_batch in enumerate(itr_valid):
                    valid_batch = self.prepare_batch(valid_batch)
                    valid_losses, valid_outputs = self.eval_on_instance(*valid_batch,
                                                                        iter=b+1,
                                                                        global_step = self.global_step)
                    for key in valid_losses:
                        this_key = 'valid/%s' % key
                        if this_key not in valid_dict:
                            valid_dict[this_key] = []
                        valid_dict[this_key].append(valid_losses[key])
                        self.logger.scalar_summary(this_key, valid_losses[key], self.global_step)

                    self.summary(valid_outputs, epoch)


                    if verbose:
                        pbar.update(1)
                        pbar.set_postfix(self._get_stats(valid_dict, 'valid'))

                if verbose:
                    pbar.close()
                # Step learning rates.
                # for sched in self.schedulers:
                #     sched.step(self.global_step)
                # Update dictionary of values.
                all_dict = train_dict
                all_dict.update(valid_dict)
                for key in all_dict:
                    all_dict[key] = np.mean(all_dict[key])
                for key in self.optim:
                    all_dict["lr_%s" % key] = \
                            self.optim[key].state_dict()['param_groups'][0]['lr']
                all_dict['time'] = time.time() - epoch_start_time
                str_ = ",".join([str(all_dict[key]) for key in all_dict])
                print(str_)
                if self.log_dir is not None:
                    if (epoch+1) == 1:
                        f.write(",".join(all_dict.keys()) + "\n")
                    f.write(str_ + "\n")
                    f.flush()
                if (epoch+1) % save_every == 0 and self.model_dir is not None:
                    self.save(filename="%s/%i.pkl" % (self.model_dir, epoch+1),
                              epoch=epoch+1)
                    #self.summary(testset, epoch=epoch+1)

                self.global_epoch += 1
        except KeyboardInterrupt:
            self.save(filename="%s/%i.pkl" % (self.model_dir, epoch + 1),
                      epoch=epoch + 1)
            print("%s/%i.pkl is saved!" % (self.model_dir, epoch + 1))
        if f is not None:
            f.close()

    def vis_batch(self, batch, outputs):
        raise NotImplementedError()

    def build_tensorboard(self, log_dir):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(log_dir)

