import os
import jittor as jt
import numpy as np
jt.flags.use_cuda = 1
from models.GAN import Generator

def main():
    checkpoint_path = './workdirs/ffhq/models/GAN_GEN_SHADOW_5_20.pkl'
    model = Generator(128, use_noise=False, mapping_layers=4)
    model.eval()
    model.load(checkpoint_path)
    latent_space_mapper = model.g_mapping
    image_generator = model.g_synthesis
    latent_size = latent_space_mapper.latent_size
    truncate_model = model.truncation
    with jt.no_grad():
        z1 = np.random.randn(latent_size,)
        z2 = np.random.randn(latent_size,)
        # z1 = np.zeros([latent_size])
        # z2 = np.ones([latent_size])
        # z2 = -z1
        z1 = (z1 / np.linalg.norm(z1)) * (latent_size ** 0.5)
        z2 = (z2 / np.linalg.norm(z2)) * (latent_size ** 0.5)
        z1 = jt.array(z1.astype(np.float32)).view(1, -1)
        z2 = jt.array(z2.astype(np.float32)).view(1, -1)
        w1 = latent_space_mapper(z1)
        w2 = latent_space_mapper(z2)
        # w2 = -w1
        # w1 = truncate_model(w1)
        # w2 = truncate_model(w2)
        beta = np.arange(0, 11) / 10
        gen_imgs = []
        for b in beta:
            w = w1 * (1.0 - b) + w2 * b
            # z = z1 * (1.0 - b) + z2 * b
            # w = latent_space_mapper(z)
            # w = truncate_model(w)
            img = image_generator(w, 5, 1.0)
            # img = adjust_dynamic_range(img)
            gen_imgs.append(img)
        gen_imgs = jt.concat(gen_imgs, dim=0)
    jt.save_image(gen_imgs, 'llerp_ffhq65k.png', nrow=gen_imgs.size(0), normalize=True, scale_each=False, pad_value=128, padding=1)

if __name__ == "__main__":
    main()