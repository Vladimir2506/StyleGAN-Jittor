import os
import random
import argparse
import glob
import numpy as np
import imageio

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=str)
    parser.add_argument('--n_imgs', type=int)
    parser.add_argument('--epochs', type=str, nargs='+')
    parser.add_argument('--out', type=str)
    args = parser.parse_args()

    n_cols = len(args.epochs)
    n_rows = args.n_imgs
    gridding_imgs = np.zeros([n_rows * 128, n_cols * 128, 3], dtype=np.uint8)
    for c in range(n_cols):
        gen_samples_fn_template = os.path.join(args.workdir, 'samples', f'gen_{args.epochs[c]}_*.png')
        gen_samples_fn = glob.glob(gen_samples_fn_template)
        random.shuffle(gen_samples_fn)
        for r in range(n_rows):
            img = imageio.imread(gen_samples_fn[r])
            ix, iy = random.randint(0, 5), random.randint(0, 5)
            ix, iy = (ix + 1) * 1 + ix * 128, (iy + 1) * 1 + iy * 128
            # breakpoint()
            gridding_imgs[r * 128:(r + 1) * 128, c * 128:(c + 1) * 128] = img[iy:iy + 128, ix:ix + 128]
    imageio.imwrite(args.out, gridding_imgs)