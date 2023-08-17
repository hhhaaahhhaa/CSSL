import argparse
import os
import glob
from tqdm import tqdm
from nnmnkwii.io import hts


def main(args):
    label_dir, output_dir = args.label_dir, args.output_dir
    os.makedirs(f"{output_dir}/TextGrid/jsut", exist_ok=True)
    for d in os.listdir(label_dir):
        if not os.path.isdir(f"{label_dir}/{d}") or d[0] == '.':
            continue
        print(f"Parsing {d}...")
        for filename in tqdm(glob.glob(f"{label_dir}/{d}/lab/*.lab")):
            dst_path = f"{output_dir}/TextGrid/jsut/{os.path.basename(filename)[:-4]}.TextGrid"
            hts.write_textgrid(dst_path, hts.load(filename))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label_dir", type=str)
    parser.add_argument("output_dir", type=str)

    args = parser.parse_args()

    main(args)
