import numpy as np
import torch


class ClassificationCollate(object):
    def __init__(self, sort=False):
        self.sort = sort

    def __call__(self, data):
        data_size = len(data)

        if self.sort:  # default no sort in this task
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
    labels = [data[idx]["label"] for idx in idxs]
    labels = np.array(labels)

    res = {
        "ids": ids,
        "labels": torch.from_numpy(labels).long(),
    }

    return res
