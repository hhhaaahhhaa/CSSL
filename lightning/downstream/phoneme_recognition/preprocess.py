import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import os
import torchaudio
from pathlib import Path

import Define
import Parsers


if Define.CUDA_LAUNCH_BLOCKING:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Preprocessor(object):
    """
    Command line interface.
    """
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.src = args.raw_dir
        self.preprocessed_root = args.preprocessed_dir
        self.processor_cls = Parsers.get_preprocessor(args.dataset)

    def exec(self, force=False):
        self.print_message()
        key_input = ""
        if not force:
            while key_input not in ["y", "Y", "n", "N"]:
                key_input = input("Proceed? ([y/n])? ")
        else:
            key_input = "y"

        if key_input in ["y", "Y"]:
            # 0. Initial features from raw data
            processor = self.processor_cls(self.src, self.preprocessed_root)
            if self.args.parse_raw:
                print("[INFO] Parsing raw corpus...")
                processor.parse_raw(n_workers=8)
            # # 1. Denoising
            # if self.args.denoise:
            #     print("[INFO] Denoising corpus...")
            #     torchaudio.set_audio_backend("sox_io")
            #     processor.denoise()
            # 2. Create Dataset
            if self.args.preprocess:
                print("[INFO] Preprocess all utterances...")
                processor.preprocess()
            if self.args.clean:
                print("[INFO] Clean...")
                processor.clean()
            if self.args.create_dataset is not None:
                print("[INFO] Creating dataset splits...")
                processor.split_dataset()

    def print_message(self):
        print("\n")
        print("------ Preprocessing ------")
        print(f"* Dataset     : {self.dataset}")
        print(f"* Raw Data path   : {self.src}")
        print(f"* Output path : {self.preprocessed_root}")
        print("\n")
        print(" [INFO] The following will be executed:")
        if self.args.parse_raw:
            print("* Parsing raw corpus")
        if self.args.denoise:
            print("* Denoising corpus")
        if self.args.preprocess:
            print("* Preprocess dataset")
        if self.args.create_dataset is not None:
            print("* Creating dataset splits")
        print("\n")


def main(args):
    Define.DEBUG = args.debug
    P = Preprocessor(args)
    P.exec(args.force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=str)
    parser.add_argument("preprocessed_dir", type=str)

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--parse_raw", action="store_true", default=False)
    parser.add_argument("--denoise", action="store_true", default=False)
    parser.add_argument("--preprocess", action="store_true", default=False)
    parser.add_argument("--clean", action="store_true", default=False)
    parser.add_argument("--create_dataset", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
