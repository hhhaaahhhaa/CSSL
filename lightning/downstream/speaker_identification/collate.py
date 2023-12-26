import numpy as np
import torch
from functools import partial


class ClassificationCollate(object):
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
        repr_info["wav"] = []
        for idx in idx_arr:
            wav = torch.from_numpy(data[idx]["wav"]).float()
            repr_info["wav"].append(wav)
        

        return (labels, repr_info)


def reprocess(data, idxs):
    """
    Pad data and calculate length of data.
    """
    ids = [data[idx]["id"] for idx in idxs]
    labels = [data[idx]["label"] for idx in idxs]
    labels = np.array(labels)

    res = {
        "ids": ids,
        "labels": torch.from_numpy(labels).long(),
    }

    return res
