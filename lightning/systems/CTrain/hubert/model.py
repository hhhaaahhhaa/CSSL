import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, List, Dict, Union

from s3prl.util.download import _urls_to_filepaths
from s3prl.upstream.hubert.expert import UpstreamExpert


class HubertCustom(UpstreamExpert):
    def __init__(self):
        ckpt = _urls_to_filepaths("https://huggingface.co/s3prl/converted_ckpts/resolve/main/hubert_base_ls960.pt")
        super().__init__(ckpt)
        self.model.feature_grad_mult = 1.0  # This enables feature extractor tuning (see forward_features())

    def _mask_forward(  # reimplement mask forward from fairseq code without nce logit calculation 
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        self._hook_hiddens.clear()  # need to remove hooks in every forard pass or else keep saving tensor and cause GPU memory leak
        features = self.model.forward_features(source)
        features = features.transpose(1, 2)
        features = self.model.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.model.forward_padding_mask(features, padding_mask)

        if self.model.post_extract_proj is not None:
            features = self.model.post_extract_proj(features)

        features = self.model.dropout_input(features)
        x, mask_indices = self.model.apply_mask(features, padding_mask, None)

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.model.encoder(
            x,
            padding_mask=padding_mask,
            layer=None,
        )

        return {"x": x, "padding_mask": padding_mask, "mask_indices": mask_indices}

    def mask_forward(self, wav_data: Union[List[torch.Tensor], List[np.ndarray]], device):
        if isinstance(wav_data[0], np.ndarray):
            wavs = [torch.from_numpy(wav).float().to(device) for wav in wav_data]
        else:
            wavs = [wav.float().to(device) for wav in wav_data]
        
        if self.task_cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        return self._mask_forward(padded_wav, wav_padding_mask)
