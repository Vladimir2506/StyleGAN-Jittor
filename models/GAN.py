import os
import datetime
import time
import timeit
import copy
import random
import numpy as np
from collections import OrderedDict

import jittor as jt
import jittor.nn as nn

import models.Losses as Losses
from data import get_data_loader
from models import update_average
from .Blocks import DiscriminatorTop, DiscriminatorBlock, InputBlock, GSynthesisBlock
from .CustomLayers import EqualizedConv2d, PixelNormLayer, EqualizedLinear, Truncation


class GMapping(nn.Module):

    def __init__(self, latent_size=512, dlatent_size=512, dlatent_broadcast=None,
                 mapping_layers=8, mapping_fmaps=512, mapping_lrmul=0.01, mapping_nonlinearity='lrelu',
                 use_wscale=True, normalize_latents=True, **kwargs):
        """
        Mapping network used in the StyleGAN paper.

        :param latent_size: Latent vector(Z) dimensionality.
        :param label_size: Label dimensionality, 0 if no labels.
        :param dlatent_size: Disentangled latent (W) dimensionality.
        :param dlatent_broadcast: Output disentangled latent (W) as [minibatch, dlatent_size]
                                  or [minibatch, dlatent_broadcast, dlatent_size].
        :param mapping_layers: Number of mapping layers.
        :param mapping_fmaps: Number of activations in the mapping layers.
        :param mapping_lrmul: Learning rate multiplier for the mapping layers.
        :param mapping_nonlinearity: Activation function: 'relu', 'lrelu'.
        :param use_wscale: Enable equalized learning rate?
        :param normalize_latents: Normalize latent vectors (Z) before feeding them to the mapping layers?
        :param kwargs: Ignore unrecognized keyword args.
        """

        super().__init__()
        
        self.latent_size = latent_size
        self.mapping_fmaps = mapping_fmaps
        self.dlatent_size = dlatent_size
        self.dlatent_broadcast = dlatent_broadcast

        # Activation function.
        act, gain = {'relu': (nn.ReLU(), np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(scale=0.2), np.sqrt(2))}[mapping_nonlinearity]

        layers = []
        # Normalize latents.
        if normalize_latents:
            layers.append(('pixel_norm', PixelNormLayer()))

        # Mapping layers. (apply_bias?)
        layers.append(('dense0', EqualizedLinear(self.latent_size, self.mapping_fmaps,
                                                 gain=gain, lrmul=mapping_lrmul, use_wscale=use_wscale)))
        layers.append(('dense0_act', act))
        for layer_idx in range(1, mapping_layers):
            fmaps_in = self.mapping_fmaps
            fmaps_out = self.dlatent_size if layer_idx == mapping_layers - 1 else self.mapping_fmaps
            layers.append(
                ('dense{:d}'.format(layer_idx),
                 EqualizedLinear(fmaps_in, fmaps_out, gain=gain, lrmul=mapping_lrmul, use_wscale=use_wscale)))
            layers.append(('dense{:d}_act'.format(layer_idx), act))

        # Output.
        self.map = nn.Sequential(OrderedDict(layers))

    def execute(self, x):
        
        x = self.map(x)
        if self.dlatent_broadcast is not None:
            x = x.unsqueeze(1).expand([x.size(0), self.dlatent_broadcast, x.size(1)])
        return x


class GSynthesis(nn.Module):

    def __init__(self, dlatent_size=512, num_channels=3, resolution=1024,
                 fmap_base=8192, fmap_decay=1.0, fmap_max=512,
                 use_styles=True, const_input_layer=True, use_noise=True, nonlinearity='lrelu',
                 use_wscale=True, use_pixel_norm=False, use_instance_norm=True, blur_filter=None,
                 structure='linear', **kwargs):
        """
        Synthesis network used in the StyleGAN paper.

        :param dlatent_size: Disentangled latent (W) dimensionality.
        :param num_channels: Number of output color channels.
        :param resolution: Output resolution.
        :param fmap_base: Overall multiplier for the number of feature maps.
        :param fmap_decay: log2 feature map reduction when doubling the resolution.
        :param fmap_max: Maximum number of feature maps in any layer.
        :param use_styles: Enable style inputs?
        :param const_input_layer: First layer is a learned constant?
        :param use_noise: Enable noise inputs?
        # :param randomize_noise: True = randomize noise inputs every time (non-deterministic),
                                  False = read noise inputs from variables.
        :param nonlinearity: Activation function: 'relu', 'lrelu'
        :param use_wscale: Enable equalized learning rate?
        :param use_pixel_norm: Enable pixel_wise feature vector normalization?
        :param use_instance_norm: Enable instance normalization?
        :param blur_filter: Low-pass filter to apply when resampling activations. None = no filtering.
        :param structure: 'fixed' = no progressive growing, 'linear' = human-readable
        :param kwargs: Ignore unrecognized keyword args.
        """

        super().__init__()

        if blur_filter is None:
            blur_filter = [1, 2, 1]

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.structure = structure

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.depth = resolution_log2 - 1

        self.num_layers = resolution_log2 * 2 - 2
        self.num_styles = self.num_layers if use_styles else 1

        act, gain = {'relu': (nn.ReLU(), np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(scale=0.2), np.sqrt(2))}[nonlinearity]

        # Early layers.
        self.init_block = InputBlock(nf(1), dlatent_size, const_input_layer, gain, use_wscale,
                                     use_noise, use_pixel_norm, use_instance_norm, use_styles, act)
        # create the ToRGB layers for various outputs
        rgb_converters = [EqualizedConv2d(nf(1), num_channels, 1, gain=1, use_wscale=use_wscale)]

        # Building blocks for remaining layers.
        blocks = []
        for res in range(3, resolution_log2 + 1):
            last_channels = nf(res - 2)
            channels = nf(res - 1)
            blocks.append(GSynthesisBlock(last_channels, channels, blur_filter, dlatent_size, gain, use_wscale,
                                          use_noise, use_pixel_norm, use_instance_norm, use_styles, act))
            rgb_converters.append(EqualizedConv2d(channels, num_channels, 1, gain=1, use_wscale=use_wscale))

        self.blocks = nn.ModuleList(blocks)
        self.to_rgb = nn.ModuleList(rgb_converters)
        self.temporaryUpsampler = lambda x: nn.interpolate(x, scale_factor=2, mode='nearest')

    def execute(self, dlatents_in, depth=0, alpha=0., labels_in=None):
        """
            forward pass of the Generator
            :param dlatents_in: Input: Disentangled latents (W) [mini_batch, num_layers, dlatent_size].
            :param labels_in:
            :param depth: current depth from where output is required
            :param alpha: value of alpha for fade-in effect
            :return: y => output
        """

        assert depth < self.depth, "Requested output depth cannot be produced"

        if self.structure == 'fixed':
            x = self.init_block(dlatents_in[:, 0:2])
            for i, block in enumerate(self.blocks):
                x = block(x, dlatents_in[:, 2 * (i + 1):2 * (i + 2)])
            images_out = self.to_rgb[-1](x)
        elif self.structure == 'linear':
            x = self.init_block(dlatents_in[:, 0:2])

            if depth > 0:
                
                for i, block in enumerate(self.blocks):
                    if i >= depth-1:
                        break
                    x = block(x, dlatents_in[:, 2 * (i + 1):2 * (i + 2)])

                residual = self.to_rgb[depth - 1](self.temporaryUpsampler(x))
                straight = self.to_rgb[depth](self.blocks[depth - 1](x, dlatents_in[:, 2 * depth:2 * (depth + 1)]))

                images_out = (alpha * straight) + ((1 - alpha) * residual)
            else:
                images_out = self.to_rgb[0](x)
        else:
            raise KeyError("Unknown structure: ", self.structure)

        return images_out


class Generator(nn.Module):

    def __init__(self, resolution, latent_size=512, dlatent_size=512,
                 truncation_psi=0.7, truncation_cutoff=8, dlatent_avg_beta=0.995,
                 style_mixing_prob=0.9, **kwargs):
        """
        # Style-based generator used in the StyleGAN paper.
        # Composed of two sub-networks (G_mapping and G_synthesis).

        :param resolution:
        :param latent_size:
        :param dlatent_size:
        :param truncation_psi: Style strength multiplier for the truncation trick. None = disable.
        :param truncation_cutoff: Number of layers for which to apply the truncation trick. None = disable.
        :param dlatent_avg_beta: Decay for tracking the moving average of W during training. None = disable.
        :param style_mixing_prob: Probability of mixing styles during training. None = disable.
        :param kwargs: Arguments for sub-networks (G_mapping and G_synthesis).
        """

        super(Generator, self).__init__()

        self.style_mixing_prob = style_mixing_prob

        # Setup components.
        self.num_layers = (int(np.log2(resolution)) - 1) * 2
        self.g_mapping = GMapping(latent_size, dlatent_size, dlatent_broadcast=self.num_layers, **kwargs)
        self.g_synthesis = GSynthesis(resolution=resolution, **kwargs)

        if truncation_psi > 0:
            self.truncation = Truncation(avg_latent=jt.zeros(dlatent_size),
                                         max_layer=truncation_cutoff,
                                         threshold=truncation_psi,
                                         beta=dlatent_avg_beta)
        else:
            self.truncation = None

    def execute(self, latents_in, depth, alpha, labels_in=None):
        """
        :param latents_in: First input: Latent vectors (Z) [mini_batch, latent_size].
        :param depth: current depth from where output is required
        :param alpha: value of alpha for fade-in effect
        :param labels_in: Second input: Conditioning labels [mini_batch, label_size].
        :return:
        """

        dlatents_in = self.g_mapping(latents_in)

        # if self.training:
        if self.is_training:
            
            if self.truncation is not None:
                self.truncation.update(dlatents_in[0, 0].detach())

            # Perform style mixing regularization.
            if self.style_mixing_prob is not None and self.style_mixing_prob > 0:
                
                latents2 = jt.random(latents_in.shape, 'float32', 'normal').stop_grad()
                dlatents2 = self.g_mapping(latents2)
                
                layer_idx = jt.array(np.arange(self.num_layers)[np.newaxis, :, np.newaxis])
                cur_layers = 2 * (depth + 1)
                mixing_cutoff = random.randint(1,
                                               cur_layers) if random.random() < self.style_mixing_prob else cur_layers
                
                mask_dlatents = layer_idx < mixing_cutoff
                dlatents_in = mask_dlatents * dlatents_in + (1-mask_dlatents)*dlatents2

        
            if self.truncation is not None:
                dlatents_in = self.truncation(dlatents_in)

        fake_images = self.g_synthesis(dlatents_in, depth, alpha)
        
        return fake_images


class Discriminator(nn.Module):

    def __init__(self, resolution, num_channels=3, fmap_base=8192, fmap_decay=1.0, fmap_max=512,
                 nonlinearity='lrelu', use_wscale=True, mbstd_group_size=4, mbstd_num_features=1,
                 blur_filter=None, structure='linear', **kwargs):
        """
        Discriminator used in the StyleGAN paper.

        :param num_channels: Number of input color channels. Overridden based on dataset.
        :param resolution: Input resolution. Overridden based on dataset.
        # label_size=0,  # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
        :param fmap_base: Overall multiplier for the number of feature maps.
        :param fmap_decay: log2 feature map reduction when doubling the resolution.
        :param fmap_max: Maximum number of feature maps in any layer.
        :param nonlinearity: Activation function: 'relu', 'lrelu'
        :param use_wscale: Enable equalized learning rate?
        :param mbstd_group_size: Group size for the mini_batch standard deviation layer, 0 = disable.
        :param mbstd_num_features: Number of features for the mini_batch standard deviation layer.
        :param blur_filter: Low-pass filter to apply when resampling activations. None = no filtering.
        :param structure: 'fixed' = no progressive growing, 'linear' = human-readable
        :param kwargs: Ignore unrecognized keyword args.
        """
        super(Discriminator, self).__init__()

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.mbstd_num_features = mbstd_num_features
        self.mbstd_group_size = mbstd_group_size
        self.structure = structure

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.depth = resolution_log2 - 1

        act, gain = {'relu': (nn.ReLU(), np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(scale=0.2), np.sqrt(2))}[nonlinearity]

        # create the remaining layers
        blocks = []
        from_rgb = []
        for res in range(resolution_log2, 2, -1):
            
            blocks.append(DiscriminatorBlock(nf(res - 1), nf(res - 2),
                                             gain=gain, use_wscale=use_wscale, activation_layer=act,
                                             blur_kernel=blur_filter))
            
            from_rgb.append(EqualizedConv2d(num_channels, nf(res - 1), kernel_size=1,
                                            gain=gain, use_wscale=use_wscale))
        self.blocks = nn.ModuleList(blocks)

        # Building the final block.
        self.final_block = DiscriminatorTop(self.mbstd_group_size, self.mbstd_num_features,
                                            in_channels=nf(2), intermediate_channels=nf(2),
                                            gain=gain, use_wscale=use_wscale, activation_layer=act)
        from_rgb.append(EqualizedConv2d(num_channels, nf(2), kernel_size=1,
                                        gain=gain, use_wscale=use_wscale))
        self.from_rgb = nn.ModuleList(from_rgb)

        self.temporaryDownsampler = nn.Pool(kernel_size=2, op='mean')

    def execute(self, images_in, depth, alpha=1., labels_in=None):
        """
        :param images_in: First input: Images [mini_batch, channel, height, width].
        :param labels_in: Second input: Labels [mini_batch, label_size].
        :param depth: current height of operation (Progressive GAN)
        :param alpha: current value of alpha for fade-in
        :return:
        """

        assert depth < self.depth, "Requested output depth cannot be produced"

        if self.structure == 'fixed':
            x = self.from_rgb[0](images_in)
            for i, block in enumerate(self.blocks):
                x = block(x)
            scores_out = self.final_block(x)
        elif self.structure == 'linear':
            if depth > 0:
                residual = self.from_rgb[self.depth - depth](self.temporaryDownsampler(images_in))
                straight = self.blocks[self.depth - depth - 1](self.from_rgb[self.depth - depth - 1](images_in))
                x = (alpha * straight) + ((1 - alpha) * residual)

                for i, block in enumerate(self.blocks):
                    if i < self.depth - depth:
                        continue 
                    x = block(x)
            else:
                x = self.from_rgb[-1](images_in)

            scores_out = self.final_block(x)
        else:
            raise KeyError("Unknown structure: ", self.structure)
        
        return scores_out


class StyleGAN:

    def __init__(self, structure, resolution, num_channels, latent_size,
                 g_args, d_args, g_opt_args, d_opt_args, loss="relativistic-hinge", drift=0.001,
                 d_repeats=1, use_ema=False, ema_decay=0.999):
        """
        Wrapper around the Generator and the Discriminator.

        :param structure: 'fixed' = no progressive growing, 'linear' = human-readable
        :param resolution: Input resolution. Overridden based on dataset.
        :param num_channels: Number of input color channels. Overridden based on dataset.
        :param latent_size: Latent size of the manifold used by the GAN
        :param g_args: Options for generator network.
        :param d_args: Options for discriminator network.
        :param g_opt_args: Options for generator optimizer.
        :param d_opt_args: Options for discriminator optimizer.
        :param loss: the loss function to be used
                     Can either be a string =>
                          ["wgan-gp", "wgan", "lsgan", "lsgan-with-sigmoid",
                          "hinge", "standard-gan" or "relativistic-hinge"]
                     Or an instance of GANLoss
        :param drift: drift penalty for the
                      (Used only if loss is wgan or wgan-gp)
        :param d_repeats: How many times the discriminator is trained per G iteration.
        :param use_ema: boolean for whether to use exponential moving averages
        :param ema_decay: value of mu for ema
        """

        # state of the object
        assert structure in ['fixed', 'linear']
        self.structure = structure
        self.depth = int(np.log2(resolution)) - 1
        self.latent_size = latent_size
        # self.device = device
        self.d_repeats = d_repeats

        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # Create the Generator and the Discriminator
        self.gen = Generator(num_channels=num_channels,
                             resolution=resolution,
                             structure=self.structure,
                             **g_args)
        self.dis = Discriminator(num_channels=num_channels,
                                 resolution=resolution,
                                 structure=self.structure,
                                 **d_args)

        # define the optimizers for the discriminator and generator
        self.__setup_gen_optim(**g_opt_args)
        self.__setup_dis_optim(**d_opt_args)

        # define the loss function used for training the GAN
        self.drift = drift
        self.loss = self.__setup_loss(loss)

        # Use of ema
        if self.use_ema:
            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)
            # updater function:
            self.ema_updater = update_average
            # initialize the gen_shadow weights equal to the weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)

    def __setup_gen_optim(self, learning_rate, beta_1, beta_2, eps):
        
        self.gen_optim = nn.Adam(self.gen.parameters(), lr=learning_rate, betas=(beta_1, beta_2), eps=eps)

    def __setup_dis_optim(self, learning_rate, beta_1, beta_2, eps):
        
        self.dis_optim = nn.Adam(self.dis.parameters(), lr=learning_rate, betas=(beta_1, beta_2), eps=eps)

    def __setup_loss(self, loss):
        if isinstance(loss, str):
            loss = loss.lower()  # lowercase the string

            if loss == "standard-gan":
                loss = Losses.StandardGAN(self.dis)
            elif loss == "hinge":
                loss = Losses.HingeGAN(self.dis)
            elif loss == "relativistic-hinge":
                loss = Losses.RelativisticAverageHingeGAN(self.dis)
            elif loss == "logistic":
                loss = Losses.LogisticGAN(self.dis)
            else:
                raise ValueError("Unknown loss function requested")

        elif not isinstance(loss, Losses.GANLoss):
            raise ValueError("loss is neither an instance of GANLoss nor a string")

        return loss

    def __progressive_down_sampling(self, real_batch, depth, alpha):
        """
        private helper for down_sampling the original images in order to facilitate the
        progressive growing of the layers.

        :param real_batch: batch of real samples
        :param depth: depth at which training is going on
        :param alpha: current value of the fade-in alpha
        :return: real_samples => modified real batch of samples
        """


        if self.structure == 'fixed':
            return real_batch

        down_sample_factor = int(np.power(2, self.depth - depth - 1))
        prior_down_sample_factor = max(int(np.power(2, self.depth - depth)), 0)

        ds_real_samples = nn.pool(real_batch,down_sample_factor,op='mean',stride=down_sample_factor)
        
        if depth > 0:
            prior_ds_real_samples = nn.interpolate(nn.pool(real_batch,prior_down_sample_factor,op='mean',stride=prior_down_sample_factor), scale_factor=2, mode='nearest')
        else:
            prior_ds_real_samples = ds_real_samples

        # real samples are a combination of ds_real_samples and prior_ds_real_samples
        real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

        # return the so computed real_samples
        return real_samples

    def optimize_discriminator(self, noise, real_batch, depth, alpha):
        """
        performs one step of weight update on discriminator using the batch of data

        :param noise: input noise of sample generation
        :param real_batch: real samples batch
        :param depth: current depth of optimization
        :param alpha: current alpha for fade-in
        :return: current loss (Wasserstein loss)
        """

        real_samples = self.__progressive_down_sampling(real_batch, depth, alpha)

        loss_val = 0
        for _ in range(self.d_repeats):
            # generate a batch of samples
            fake_samples = self.gen(noise, depth, alpha).detach()

            loss = self.loss.dis_loss(real_samples, fake_samples, depth, alpha)
           
            loss.sync()
           
            self.dis_optim.step(loss)

            loss_val += loss.item()

        return loss_val / self.d_repeats

    def optimize_generator(self, noise, real_batch, depth, alpha):
        """
        performs one step of weight update on generator for the given batch_size

        :param noise: input random noise required for generating samples
        :param real_batch: batch of real samples
        :param depth: depth of the network at which optimization is done
        :param alpha: value of alpha for fade-in effect
        :return: current loss (Wasserstein estimate)
        """

        real_samples = self.__progressive_down_sampling(real_batch, depth, alpha)

        # generate fake samples:
        fake_samples = self.gen(noise, depth, alpha)

        # Change this implementation for making it compatible for relativisticGAN
        loss = self.loss.gen_loss(real_samples, fake_samples, depth, alpha)

        
        loss.sync()
        
        self.gen_optim.step(loss)

        if self.use_ema:
            self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)

        # return the loss value
        return loss.item()

    @staticmethod
    def create_grid(samples, scale_factor, img_file):
        if scale_factor > 1:
            samples = nn.interpolate(samples, scale_factor=scale_factor, mode='nearest')
        jt.save_image(samples, img_file, nrow=int(np.sqrt(len(samples))), normalize=True, scale_each=False, pad_value=128, padding=1)

    def train(self, dataset, num_workers, epochs, batch_sizes, fade_in_percentage, logger, output,
              num_samples=36, start_depth=0, feedback_factor=100, checkpoint_factor=1):
        """
        Utility method for training the GAN. Note that you don't have to necessarily use this
        you can use the optimize_generator and optimize_discriminator for your own training routine.

        :param dataset: object of the dataset used for training.
                        Note that this is not the data loader (we create data loader in this method
                        since the batch_sizes for resolutions can be different)
        :param num_workers: number of workers for reading the data. def=3
        :param epochs: list of number of epochs to train the network for every resolution
        :param batch_sizes: list of batch_sizes for every resolution
        :param fade_in_percentage: list of percentages of epochs per resolution used for fading in the new layer
                                   not used for first resolution, but dummy value still needed.
        :param logger:
        :param output: Output dir for samples,models,and log.
        :param num_samples: number of samples generated in sample_sheet. def=36
        :param start_depth: start training from this depth. def=0
        :param feedback_factor: number of logs per epoch. def=100
        :param checkpoint_factor:
        :return: None (Writes multiple files to disk)
        """

        assert self.depth <= len(epochs), "epochs not compatible with depth"
        assert self.depth <= len(batch_sizes), "batch_sizes not compatible with depth"
        assert self.depth <= len(fade_in_percentage), "fade_in_percentage not compatible with depth"

        # turn the generator and discriminator into train mode
        self.gen.train()
        self.dis.train()
        if self.use_ema:
            self.gen_shadow.train()

        # create a global time counter
        global_time = time.time()

        # create fixed_input for debugging
        fixed_input = jt.random([num_samples, self.latent_size], 'float32', 'normal').stop_grad()

        # config depend on structure
        logger.info("Starting the training process ... \n")
        if self.structure == 'fixed':
            start_depth = self.depth - 1
        step = 1  # counter for number of iterations
        for current_depth in range(start_depth, self.depth):
            current_res = np.power(2, current_depth + 2)
            logger.info("Currently working on depth: %d", current_depth + 1)
            logger.info("Current resolution: %d x %d" % (current_res, current_res))

            ticker = 1
            data = get_data_loader(dataset, batch_sizes[current_depth], num_workers)

            for epoch in range(1, epochs[current_depth] + 1):
                start = timeit.default_timer()  # record time at the start of epoch

                logger.info("Epoch: [%d]" % epoch)
                # total_batches = len(iter(data))
                total_batches = len(data)

                fade_point = int((fade_in_percentage[current_depth] / 100)
                                 * epochs[current_depth] * total_batches)

                for i, (batch, useless) in enumerate(data, 1):
                    # calculate the alpha for fading in the layers
                    alpha = ticker / fade_point if ticker <= fade_point else 1

                    # extract current batch of data for training
                    images = batch
                    gan_input = jt.random([images.shape[0], self.latent_size], 'float32', 'normal').stop_grad()

                    # optimize the discriminator:
                    dis_loss = self.optimize_discriminator(gan_input, images, current_depth, alpha)

                    # optimize the generator:
                    gen_loss = self.optimize_generator(gan_input, images, current_depth, alpha)

                    # provide a loss feedback
                    if i % int(total_batches / feedback_factor + 1) == 0 or i == 1:
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]
                        logger.info(
                            "Elapsed: [%s] Step: %d  Batch: %d  D_Loss: %f  G_Loss: %f"
                            % (elapsed, step, i, dis_loss, gen_loss))

                        # create a grid of samples and save it
                        os.makedirs(os.path.join(output, 'samples'), exist_ok=True)
                        gen_img_file = os.path.join(output, 'samples', "gen_" + str(current_depth)
                                                    + "_" + str(epoch) + "_" + str(i) + ".png")

                        # with torch.no_grad():
                        with jt.no_grad():
                            self.create_grid(
                                samples=self.gen(fixed_input, current_depth, alpha).detach() if not self.use_ema
                                else self.gen_shadow(fixed_input, current_depth, alpha).detach(),
                                scale_factor=int(
                                    np.power(2, self.depth - current_depth - 1)) if self.structure == 'linear' else 1,
                                img_file=gen_img_file,
                            )

                    # increment the alpha ticker and the step
                    ticker += 1
                    step += 1

                elapsed = timeit.default_timer() - start
                elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]
                logger.info("Time taken for epoch: %s\n" % elapsed)

                if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == epochs[current_depth]:
                    save_dir = os.path.join(output, 'models')
                    os.makedirs(save_dir, exist_ok=True)
                    gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(current_depth) + "_" + str(epoch) + ".pkl")
                    dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(current_depth) + "_" + str(epoch) + ".pkl")
                    
                    self.gen.save(gen_save_file)
                    logger.info("Saving the model to: %s\n" % gen_save_file)
                    self.dis.save(dis_save_file)

                    # also save the shadow generator if use_ema is True
                    if self.use_ema:
                        gen_shadow_save_file = os.path.join(
                            save_dir, "GAN_GEN_SHADOW_" + str(current_depth) + "_" + str(epoch) + ".pkl")
                        self.gen_shadow.save(gen_shadow_save_file)
                        logger.info("Saving the model to: %s\n" % gen_shadow_save_file)

        logger.info('Training completed.\n')

