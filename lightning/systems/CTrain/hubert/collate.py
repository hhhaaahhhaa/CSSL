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
        # labels = reprocess_mu(data, idx_arr)  # TODO: Support mix units

        repr_info = {}
        repr_info["wav"] = [torch.from_numpy(data[idx]["wav"]).float() for idx in idx_arr]

        return (labels, repr_info)


def reprocess(data, idxs):
    """
    Pad data and calculate length of data.
    """
    ids = [data[idx]["id"] for idx in idxs]
    # speakers = [data[idx]["speaker"] for idx in idxs]
    # speakers = np.array(speakers)

    # Code labels
    codes = [data[idx]["idxs"] for idx in idxs]
    code_lens = np.array([code.shape[0] for code in codes])
    codes = pad_1D(codes)

    res = {
        "ids": ids,
        "codes": torch.from_numpy(codes).long(),
        "code_lens": torch.from_numpy(code_lens),
        "max_code_len": max(code_lens),
    }

    return res


def reprocess_mu(data, idxs):  # TODO: Support mix units
    """
    reprocess for mix unit training
    """
    raise NotImplementedError
