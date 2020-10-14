# import tensorflow as tf
from tensorboardX import SummaryWriter
from plotting_utils import plot_spectrogram_to_numpy

class Logger(object):
    """Using tensorboardX such that need no dependency on tensorflow."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, image, step):
        self.writer.add_image(
            tag,
            image,
            #plot_spectrogram_to_numpy(image.T),
            step,
            dataformats='HWC')

    def dist_summary(self, tag, weights, step):
        # # plot distribution of parameters
        # for tag, value in model.named_parameters():
        #     tag = tag.replace('.', '/')
        #     self.add_histogram(tag, value.data.cpu().numpy(), iteration)
        self.writer.add_histogram(tag, weights, step)