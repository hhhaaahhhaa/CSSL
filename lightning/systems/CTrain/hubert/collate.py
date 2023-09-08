import numpy as np
import torch
from functools import partial

from lightning.utils.tool import pad_1D


class Collate(object):
    def __init__(self):
        pass

    def collate_fn(self, sort=False):
        return partial(self._collate_fn, sort=sort)

    def _collate_fn(self, data, sort=False):
        data_size = len(data)

        if sort:  # default no sort in this task
            # len_arr = np.array([d["duration"].shape[0] for d in data])
            # idx_arr = np.argsort(-len_arr)
            raise NotImplementedError
        else:
            idx_arr = np.arange(data_size)
        labels = reprocess(data, idx_arr)

        repr_info = {}
        repr_info["wav"] = [torch.from_numpy(data[idx]["wav"]).float() for idx in idx_arr]

        return (labels, repr_info)


def reprocess(data, idxs):
    """
    Pad data and calculate length of data.
    """
    ids = [data[idx]["id"] for idx in idxs]
    speakers = [data[idx]["speaker"] for idx in idxs]
    speakers = np.array(speakers)

    # Text labels
    clusters = [data[idx]["idxs"] for idx in idxs]
    cluster_lens = np.array([cluster.shape[0] for cluster in clusters])
    clusters = pad_1D(clusters)

    res = {
        "ids": ids,
        "speaker_args": torch.from_numpy(speakers).long(),
        "clusters": torch.from_numpy(clusters).long(),
        "cluster_lens": torch.from_numpy(cluster_lens),
        "max_cluster_len": max(cluster_lens),
    }

    return res
