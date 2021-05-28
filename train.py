import os
import argparse
import shutil
import jittor as jt

from utils import make_logger, list_dir_recursively_with_ignore, copy_files_and_create_dirs
from models.GAN import StyleGAN

from config import cfg as opt

if __name__ == '__main__':

    jt.flags.use_cuda = 1
    jt.flags.log_silent = 1

    parser = argparse.ArgumentParser(description="StyleGAN jittor implementation.")
    parser.add_argument('--config', default='./configs/sample.yaml')
    parser.add_argument("--start_depth", action="store", type=int, default=0, help="Starting depth for training the network")
    args = parser.parse_args()

    opt.merge_from_file(args.config)
    opt.freeze()

    output_dir = opt.output_dir
    os.makedirs(output_dir, exist_ok=True)

    files = list_dir_recursively_with_ignore('.', ignores=['diagrams', 'configs', 'workdirs'])
    files = [(f[0], os.path.join(output_dir, "src", f[1])) for f in files]
    copy_files_and_create_dirs(files)
    shutil.copy2(args.config, output_dir)

    # logger
    logger = make_logger("project", opt.output_dir, 'log')

    StyleGAN(structure=opt.structure,
             resolution=opt.dataset.resolution,
             num_channels=opt.dataset.channels,
             latent_size=opt.model.gen.latent_size,
             g_args=opt.model.gen,
             d_args=opt.model.dis,
             g_opt_args=opt.model.g_optim,
             d_opt_args=opt.model.d_optim,
             loss=opt.loss,
             drift=opt.drift,
             d_repeats=opt.d_repeats,
             use_ema=opt.use_ema,
             ema_decay=opt.ema_decay
             ).train(
                 dataset=opt.dataset,
                 num_workers=opt.num_works,
                 epochs=opt.sched.epochs,
                 batch_sizes=opt.sched.batch_sizes,
                 fade_in_percentage=opt.sched.fade_in_percentage,
                 logger=logger,
                 output=output_dir,
                 num_samples=opt.num_samples,
                 start_depth=args.start_depth,
                 feedback_factor=opt.feedback_factor,
                 checkpoint_factor=opt.checkpoint_factor
                 )
