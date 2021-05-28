import argparse
import math
import jittor as jt
import numpy as np
from PIL import Image

from config import cfg as opt
from models.GAN import Generator

def adjust_dynamic_range(data, drange_in=(-1, 1), drange_out=(0, 1)):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return jt.clamp(data, min_v=0, max_v=1)

def draw_style_mixing_figure(png, gen, out_depth, src_seeds, dst_seeds, style_ranges):
    
    n_col = len(src_seeds)
    n_row = len(dst_seeds)
    w = h = 2 ** (out_depth + 2)
    latent_size = gen.g_mapping.latent_size
    z1s = []
    for seed in src_seeds:
        z1 = np.random.RandomState(seed).randn(latent_size, )
        z1 = (z1 / np.linalg.norm(z1)) * (latent_size ** 0.5)
        z1s.append(z1)
    z2s = []
    for seed in dst_seeds:
        z2 = np.random.RandomState(seed).randn(latent_size, )
        z2 = (z2 / np.linalg.norm(z2)) * (latent_size ** 0.5)
        z2s.append(z2)
    src_latents_np = np.stack(z1s)
    dst_latents_np = np.stack(z2s)
    # breakpoint()
    src_latents = jt.array(src_latents_np.astype(np.float32))
    dst_latents = jt.array(dst_latents_np.astype(np.float32))
    src_dlatents = gen.truncation(gen.g_mapping(src_latents))
    # src_dlatents = gen.g_mapping(src_latents)
    dst_dlatents = gen.truncation(gen.g_mapping(dst_latents))
    # dst_dlatents = gen.g_mapping(dst_latents)
    src_images = gen.g_synthesis(src_dlatents, depth=out_depth, alpha=1)
    dst_images = gen.g_synthesis(dst_dlatents, depth=out_depth, alpha=1)
    src_dlatents_np = src_dlatents.data
    dst_dlatents_np = dst_dlatents.data
    canvas = Image.new('RGB', (w * (n_col + 1), h * (n_row + 1)), 'white')
    for col, src_image in enumerate(list(src_images)):
        src_image = adjust_dynamic_range(src_image)
        src_image = src_image.multiply(255).clamp(0, 255).permute(1, 2, 0).data.astype(np.uint8)
        canvas.paste(Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
    for row, dst_image in enumerate(list(dst_images)):
        dst_image = adjust_dynamic_range(dst_image)
        dst_image = dst_image.multiply(255).clamp(0, 255).permute(1, 2, 0).data.astype(np.uint8)
        canvas.paste(Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * h))
        row_dlatents = np.stack([dst_dlatents_np[row]] * n_col)
        row_dlatents[:, style_ranges[row]] = src_dlatents_np[:, style_ranges[row]]
        row_dlatents = jt.array(row_dlatents)
        row_images = gen.g_synthesis(row_dlatents, depth=out_depth, alpha=1)
        for col, image in enumerate(list(row_images)):
            image = adjust_dynamic_range(image)
            image = image.multiply(255).clamp(0, 255).permute(1, 2, 0).data.astype(np.uint8)
            canvas.paste(Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
    canvas.save(png)

if __name__ == "__main__":

    jt.flags.use_cuda = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument('--src_seeds', type=int, nargs='+')
    parser.add_argument('--dst_seed', type=int)
    parser.add_argument('--out', type=str)
    args = parser.parse_args()
    opt.merge_from_file(args.config)
    opt.model.gen.use_noise = False
    generator = Generator(resolution=opt.dataset.resolution, num_channels=opt.dataset.channels, structure=opt.structure, **opt.model.gen)
    generator.load(args.model)
    generator.eval()
    # srs = [
    #     range(0, 0),
    #     range(0, 2),
    #     range(0, 4),
    #     range(0, 6),
    #     range(0, 8),
    #     range(0, 10),
    #     range(0, 12)
    # ]
    # srs = list(reversed(srs))
    srs = [
        range(0, 12),
        range(2, 12),
        range(4, 12),
        range(6, 12),
        range(8, 12),
        range(10, 12),
        range(12, 12)
    ]
    with jt.no_grad():
        draw_style_mixing_figure(
            args.out, generator, out_depth=generator.g_synthesis.depth - 1, 
            src_seeds=args.src_seeds, dst_seeds=[args.dst_seed] * 7, style_ranges=srs
        )
