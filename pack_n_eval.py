import numpy as np
import math
from evaluator import metric
import argparse
from tqdm import tqdm
from PIL import Image

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

parser = argparse.ArgumentParser()
parser.add_argument("--sample_folder_dir", type=str)
parser.add_argument("--refer_file_name", type=str, default='/mnt/workspace/diffusion/evaluations/VIRTUAL_imagenet256_labeled.npz')
args = parser.parse_args()

sample_file_name = create_npz_from_sample_folder(args.sample_folder_dir, 50176)
results = metric(args.refer_file_name, sample_file_name)
with open('{}.txt'.format(args.sample_folder_dir), 'w') as f:
    for key in results:
        f.write(key+':'+' '+str(results[key])+'\n')
print("Done.")
# metric(args.refer_file_name, '/mnt/workspace/diffusion/fast-DiT/samples/DiT-XL-2-0850000-size-256-vae-ema-cfg-1.5-seed-0.npz')