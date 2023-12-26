import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pylab as plt


class AttentionVisualizer(object):
    def __init__(self, figsize=(32, 16)):
        self.figsize = figsize

    def plot(self, info):
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)
        cax = ax.matshow(info["attn"])
        fig.colorbar(cax, ax=ax)

        ax.set_title(info["title"], fontsize=28)
        ax.set_xticks(np.arange(len(info["x_labels"])))
        ax.set_xticklabels(info["x_labels"], rotation=90, fontsize=8)
        ax.set_yticks(np.arange(len(info["y_labels"])))
        ax.set_yticklabels(info["y_labels"], fontsize=8)

        return fig
