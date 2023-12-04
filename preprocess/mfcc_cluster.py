import argparse
import os
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
import random
import time
import faiss
from typing import Any, List

from dlhlp_lib.utils.numeric import numpy_exist_nan

import Define
from lightning.systems.CTrain.hubert.Parsers import get_parser


SEED = 666


class MfccFeatureReader(object):
    """ Copied from fairseq/hubert """
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def get_feats(self, x: np.ndarray):
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            x = x.view(1, -1)

            mfccs = torchaudio.compliance.kaldi.mfcc(
                waveform=x,
                sample_frequency=self.sample_rate,
                use_energy=False,
                frame_shift=20.0,
            )  # (time, freq)
            mfccs = mfccs.transpose(0, 1)  # (freq, time)
            deltas = torchaudio.functional.compute_deltas(mfccs)
            ddeltas = torchaudio.functional.compute_deltas(deltas)
            concat = torch.cat([mfccs, deltas, ddeltas], dim=0)
            concat = concat.transpose(0, 1).contiguous()  # (freq, time)
            return concat
        
    def __call__(self, x):
        return self.get_feats(x).numpy()


def get_mfcc_centroids(
    model: MfccFeatureReader,
    parser_name: str,
    roots: List[str],
    n_cluster: int,
    n_sample: int=1024
) -> np.ndarray:
    for root in roots:
        data_parser = get_parser(parser_name)(root)
        queries = data_parser.get_all_queries()
        queries = random.sample(queries, n_sample)

        all_frames = []
        for query in tqdm(queries):
            try:
                wav = data_parser.wav_16000.read_from_query(query)
                with torch.no_grad():
                    repr = model(wav)
                    assert not numpy_exist_nan(repr)
                    all_frames.append(repr)
            except:
                continue

    # Concatenate and perform KMeans clustering.
    all_frames = np.concatenate(all_frames, axis=0)

    st = time.time()
    print("Perform KMeans...")
    kmeans = faiss.Kmeans(d=all_frames.shape[1], k=n_cluster, verbose=True, seed=SEED)
    kmeans.train(all_frames)
    print(f"Done in {time.time()-st:.2f}s. Data {all_frames.shape} => Centroids {kmeans.centroids.shape}.")

    return kmeans.centroids


def main(args):
    model = MfccFeatureReader(sample_rate=16000)
    centroids = get_mfcc_centroids(model, args.parser_name, args.roots, args.n_cluster, args.n_sample)

    for root in args.roots:
        data_parser = get_parser(args.parser_name)(root)
        data_parser.create_unit_feature(unit_name=args.cluster_name)
        queries = data_parser.get_all_queries()
        fail_cnt = 0
        for query in tqdm(queries):
            try:
                with torch.no_grad():
                    wav = data_parser.wav_16000.read_from_query(query)
                    repr = model(wav)
                    assert not numpy_exist_nan(repr)
                    distance = np.linalg.norm(np.expand_dims(repr, axis=1) - np.expand_dims(centroids, axis=0), axis=2)  # L, n_c
                    cluster_ids = np.argmin(distance, axis=1)
                    cluster_ids = [str(x) for x in cluster_ids]
                    data_parser.units[args.cluster_name].codes.save(" ".join(cluster_ids), query)
            except:
                fail_cnt += 1
                print("MFCC extraction fails in query:")
                print(query)
                continue
        print("Skipped: ", fail_cnt)


def get_preprocess_args():
    
    parser = argparse.ArgumentParser(description='preprocess arguments for any dataset.')

    parser.add_argument('-d', '--roots', type=str, nargs='+', help="preprocessed data directories")
    parser.add_argument('-n', '--cluster_name', type=str, help="cluster identifier")
    parser.add_argument('-p', '--parser_name', default="plain", type=str, help="paresr identifier")
    parser.add_argument('--n_cluster', type=int, default=512)
    parser.add_argument('--n_sample', type=int, default=1024)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    random.seed(SEED)
    if Define.CUDA_LAUNCH_BLOCKING:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = get_preprocess_args()
    main(args)
