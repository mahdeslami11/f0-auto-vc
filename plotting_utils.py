import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np

def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def plot_spectrogram_to_numpy(title, spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.title(title)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

def plot_f0_to_numpy(title, f0):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.plot(range(len(f0)), f0, color='green')
    plt.title(title)
    plt.xlabel("Frames")
    plt.ylabel("F0")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

def plot_f0_outputs_to_numpy(title, f0_target, f0_predicted):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(range(len(f0_target)), f0_target, alpha=0.5,
            color='green', label='target')
    ax.plot(range(len(f0_predicted)), f0_predicted, alpha=0.5,
            color='red', label='predicted')

    plt.title(title)
    plt.xlabel("Frames (Green target, Red predicted)")
    plt.ylabel("F0")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data