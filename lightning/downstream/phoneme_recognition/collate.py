import numpy as np
import torch
from functools import partial

from lightning.utils.tool import pad_1D, pad_2D


class PRCollate(object):
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
        repr_info["n_symbols"] = data[0]["n_symbols"]
        repr_info["lang_id"] = data[0]["lang_id"]

        return (labels, repr_info)


def reprocess(data, idxs):
    """
    Pad data and calculate length of data.
    """
    ids = [data[idx]["id"] for idx in idxs]
    speakers = [data[idx]["speaker"] for idx in idxs]
    speakers = np.array(speakers)

    # Text labels
    raw_texts = [data[idx]["raw_text"] for idx in idxs]
    texts = [data[idx]["text"] for idx in idxs]
    text_lens = np.array([text.shape[0] for text in texts])
    texts = pad_1D(texts)

    res = {
        "ids": ids,
        "speaker_args": torch.from_numpy(speakers).long(),
        "raw_texts": raw_texts,
        "texts": torch.from_numpy(texts).long(),
        "text_lens": torch.from_numpy(text_lens),
        "max_text_len": max(text_lens),
    }

    # Framewise labels
    if "expanded_text" in data[0]:
        expanded_texts = [data[idx]["expanded_text"] for idx in idxs]
        expanded_text_lens = np.array([expanded_text.shape[0] for expanded_text in expanded_texts])
        expanded_texts = pad_1D(expanded_texts)
        res.update({
            "expanded_texts": torch.from_numpy(expanded_texts).long(),
            "expanded_text_lens": torch.from_numpy(expanded_text_lens),
            "max_expanded_text_len": max(expanded_text_lens),
        })
        
    return res
