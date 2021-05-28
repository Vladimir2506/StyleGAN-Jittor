# CUDA_VISIBLE_DEVICES=0 \
# python latent_gate_interp.py \
# --config /mnt/data1/stylegan/workdirs/colorchar128/colorchar.yaml \
# --model /mnt/data1/stylegan/workdirs/colorchar128/models/GAN_GEN_5_240.pkl \
# --src_seeds 7422 9914 23 345 195 \
# --dst_seed 1129 \
# --out colorchar128_latent_interp.png

CUDA_VISIBLE_DEVICES=1 \
python latent_gate_interp.py \
--config /mnt/data1/stylegan/workdirs/ffhq/ffhq.yaml \
--model /mnt/data1/stylegan/workdirs/ffhq/models/GAN_GEN_5_20.pkl \
--src_seeds 7422 9914 23 345 195 \
--dst_seed 7933 \
--out ffhq_latent_interp.png