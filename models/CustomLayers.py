import jittor as jt 
import jittor.nn as nn 
import numpy as np
from collections import OrderedDict

class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def execute(self, x):
        return x / jt.sqrt(jt.mean(x.sqr(), dim=1, keepdims=True) + self.epsilon)
    
class Upscale2d(nn.Module):
    @staticmethod
    def upscale2d(x, factor=2, gain=1):
        assert x.ndim == 4
        if gain != 1:
            x = x * gain
        if factor != 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand([shape[0], shape[1], shape[2], factor, shape[3], factor])
            x = x.view(shape[0], shape[1], factor * shape[2], factor * shape[3])
        return x

    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    def execute(self, x):
        return self.upscale2d(x, factor=self.factor, gain=self.gain)

class BlurLayer(nn.Module):
    
    def __init__(self, kernel=None, normalize=True, flip=False, stride=1):
        
        super().__init__()
        # breakpoint()
        if kernel is None:
            kernel = [1, 2, 1]
        kernel = jt.float32(kernel)
        kernel = kernel.unsqueeze(dim=1) * kernel.unsqueeze(dim=0)
        kernel = kernel.unsqueeze(dim=0).unsqueeze(dim=0)
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.kernel = kernel
        
        self.stride = stride

    def execute(self, x):
        
        size = self.kernel.shape
        kernel = self.kernel.expand([x.size(1), size[1], size[2], size[3]])
        x = nn.conv2d(
            x, 
            kernel, 
            stride=self.stride, 
            padding=int((size[2]-1)/2),
            groups=x.size(1)
            )
        return x

class Downscale2d(nn.Module):

    def __init__(self, factor=2, gain=1):

        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor
        self.gain = gain
        if factor == 2:
            f = [np.sqrt(gain) / factor] * factor
            self.blur = BlurLayer(kernel=f, normalize=False, stride=factor)
        else:
            self.blur = None

    def execute(self, x):
        assert x.ndim == 4
        if self.blur is not None and x.dtype == jt.float32:
            return self.blur(x)

        # Apply gain.
        if self.gain != 1:
            x = x * self.gain

        # No-op => early exit.
        if self.factor == 1:
            return x

        return nn.pool(x, kernel_size=self.factor, op='mean', stride=self.factor)

class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_size, output_size, gain=2 ** 0.5, use_wscale=False, lrmul=1, bias=True):
        super().__init__()

        self.use_bias_ = bias
        he_std = gain * input_size ** (-0.5)  # He init
        
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.weight = nn.init.gauss([output_size, input_size], 'float32', std=init_std)
        if self.use_bias_:
            self.bias = nn.init.constant([output_size], 'float32', 0.0)
            self.b_mul = lrmul
        else:
            self.bias = None

    def execute(self, x):
        
        y = nn.matmul_transpose(x, self.weight * self.w_mul)
        if self.use_bias_:
            y += self.bias * self.b_mul
        return y
    
class EqualizedConv2d(nn.Module):
    """Conv layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_channels, output_channels, kernel_size, stride=1, gain=2 ** 0.5, use_wscale=False, lrmul=1, bias=True, intermediate=None, upscale=False, downscale=False):
        super().__init__()
        self.use_bias_ = bias
        self.upscale_ = upscale
        self.downscale_ = downscale

        if upscale:
            self.upscale = Upscale2d()
        else:
            self.upscale = None

        if downscale:
            self.downscale = Downscale2d()
        else:
            self.downscale = None

        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5) 
        self.kernel_size = kernel_size
        
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
       
        self.weight = nn.init.gauss([output_channels, input_channels, kernel_size, kernel_size], 'float32', std=init_std)
        if self.use_bias_:
            self.bias = nn.init.constant([output_channels], 'float32', 0.0)
            self.b_mul = lrmul
        else:
            self.bias = None
        self.intermediate = intermediate

    def execute(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul

        have_convolution = False
        if self.upscale is not None and min(x.shape[2:]) * 2 >= 128:
            # this is the fused upscale + conv from StyleGAN, sadly this seems incompatible with the non-fused way
            # this really needs to be cleaned up and go into the conv...
            w = self.weight * self.w_mul
            w = w.permute(1, 0, 2, 3)
            # probably applying a conv on w would be more efficient. also this quadruples the weight (average)?!
            # w = F.pad(w, [1, 1, 1, 1])
            w = nn.pad(w, [1,1,1,1])
            w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            x = nn.conv_transpose2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            have_convolution = True
        elif self.upscale is not None:
            x = self.upscale(x)

        downscale = self.downscale
        intermediate = self.intermediate
        if downscale is not None and min(x.shape[2:]) >= 128:
            w = self.weight * self.w_mul
            # w = F.pad(w, [1, 1, 1, 1])
            w = nn.pad(w, [1, 1, 1, 1])
            # in contrast to upscale, this is a mean...
            w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) * 0.25  # avg_pool?
            # x = F.conv2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            x = nn.conv2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            have_convolution = True
            downscale = None
        elif downscale is not None:
            assert intermediate is None
            intermediate = downscale

        if not have_convolution and intermediate is None:
            # return F.conv2d(x, self.weight * self.w_mul, bias, padding=self.kernel_size // 2)
            return nn.conv2d(x, self.weight * self.w_mul, bias, padding=self.kernel_size // 2)
        elif not have_convolution:
            # x = F.conv2d(x, self.weight * self.w_mul, None, padding=self.kernel_size // 2)
            x = nn.conv2d(x, self.weight * self.w_mul, None, padding=self.kernel_size // 2)

        if intermediate is not None:
            x = intermediate(x)

        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
        return x

class NoiseLayer(nn.Module):
    """adds noise. noise is per pixel (constant over channels) with per-channel weight"""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.init.constant([channels], 'float32', 0.0)
        self.noise = None

    def execute(self, x, noise=None):
        if noise is None and self.noise is None:
            # noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            # TODO: device
            noise = jt.random([x.size(0), 1, x.size(2), x.size(3)], x.dtype, 'normal')
        elif noise is None:
            # here is a little trick: if you get all the noise layers and set each
            # modules .noise attribute, you can have pre-defined noise.
            # Very useful for analysis
            noise = self.noise
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x

class StyleMod(nn.Module):
    def __init__(self, latent_size, channels, use_wscale):
        super(StyleMod, self).__init__()
        self.lin = EqualizedLinear(latent_size,
                                   channels * 2,
                                   gain=1.0, use_wscale=use_wscale)

    def execute(self, x, latent):
        style = self.lin(latent)  # style => [batch_size, n_channels*2]

        shape = [-1, 2, x.size(1)] + (x.ndim - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


class LayerEpilogue(nn.Module):
    """Things to do at the end of each layer."""

    def __init__(self, channels, dlatent_size, use_wscale,
                 use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer):
        super().__init__()

        layers = []
        if use_noise:
            layers.append(('noise', NoiseLayer(channels)))
        layers.append(('activation', activation_layer))
        if use_pixel_norm:
            layers.append(('pixel_norm', PixelNormLayer()))
        if use_instance_norm:
            layers.append(('instance_norm', nn.InstanceNorm2d(channels, affine=False)))

        self.top_epi = nn.Sequential(OrderedDict(layers))

        if use_styles:
            self.style_mod = StyleMod(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def execute(self, x, dlatents_in_slice=None):
        x = self.top_epi(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)
        else:
            assert dlatents_in_slice is None
        return x

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def execute(self, x):
        return x.view(x.size(0), *self.shape)

class StddevLayer(nn.Module):
    def __init__(self, group_size=4, num_new_features=1):
        super().__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features

    def execute(self, x):
        b, c, h, w = x.shape
        group_size = min(self.group_size, b)
        y = x.reshape([group_size, -1, self.num_new_features, c // self.num_new_features, h, w])
        y = y - y.mean(0, keepdims=True)
        y = (y.sqr()).mean(0, keepdims=True)
        y = (y + 1e-8).pow(0.5)
        y = y.mean([3, 4, 5], keepdims=True).squeeze(3)  # don't keep the meaned-out channels
        y = y.expand([group_size, y.size(1), y.size(2), h, w]).clone().reshape(b, self.num_new_features, h, w)
        z = jt.concat([x, y], dim=1)
        return z

class Truncation(nn.Module):
    def __init__(self, avg_latent, max_layer=8, threshold=0.7, beta=0.995):
        super().__init__()
        self.max_layer = max_layer
        self.threshold = threshold
        self.beta = beta
        
        self.avg_latent = avg_latent
        self.avg_latent.stop_grad()

    def update(self, last_avg):
        
        self.avg_latent.update(self.beta * self.avg_latent + (1. - self.beta) * last_avg)
        

    def execute(self, x):

        assert x.ndim == 3
        interp = self.avg_latent + self.threshold * (x - self.avg_latent)
        do_trunc = (jt.arange(x.size(1)) < self.max_layer).view(1, -1, 1)
        return do_trunc * interp + (1 - do_trunc) * x


