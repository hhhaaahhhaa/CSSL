import pytorch_lightning as pl

from dlhlp_lib.s3prl import S3PRLExtractor


class S3PRLWrapper(pl.LightningModule):
    def __init__(self, s3prl_name: str) -> None:
        super().__init__()
        self.extractor = S3PRLExtractor(s3prl_name)
        self.n_layers = self.extractor.n_layers
        self.dim = self.extractor.dim

    def forward(self, x):
        return self.extractor.extract(x)
    