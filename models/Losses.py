import jittor as jt 
import jittor.nn as nn

class GANLoss:
    """ Base class for all losses

        @args:
        dis: Discriminator used for calculating the loss
             Note this must be a part of the GAN framework
    """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")


class ConditionalGANLoss:
    """ Base class for all conditional losses """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, labels, height, alpha):
        raise NotImplementedError("gen_loss method has not been implemented")


# =============================================================
# Normal versions of the Losses:
# =============================================================

class StandardGAN(GANLoss):

    def __init__(self, dis):
        from jittor.nn import BCEWithLogitsLoss

        super().__init__(dis)

        # define the criterion and activation used for object
        self.criterion = BCEWithLogitsLoss()

    def dis_loss(self, real_samps, fake_samps, height, alpha):
      
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)
        real_loss = self.criterion(jt.squeeze(r_preds), jt.ones(real_samps.shape[0]))
        fake_loss = self.criterion(jt.squeeze(f_preds), jt.zeros(fake_samps.shape[0]))

        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps, height, alpha):
        preds, _, _ = self.dis(fake_samps, height, alpha)
        return self.criterion(jt.squeeze(preds), jt.ones(fake_samps.shape[0]))


class HingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        loss = (jt.mean(nn.relu(1 - r_preds)) + jt.mean(nn.relu(1 + f_preds)))
        return loss

    def gen_loss(self, _, fake_samps, height, alpha):
       
        return -jt.mean(self.dis(fake_samps, height, alpha))


class RelativisticAverageHingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        # Obtain predictions
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        r_f_diff = r_preds - jt.mean(f_preds)

        f_r_diff = f_preds - jt.mean(r_preds)

        loss = (jt.mean(nn.relu(1 - r_f_diff)) + jt.mean(nn.relu(1 + f_r_diff)))

        return loss

    def gen_loss(self, real_samps, fake_samps, height, alpha):
        # Obtain predictions
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)
        r_f_diff = r_preds - jt.mean(f_preds)
        f_r_diff = f_preds - jt.mean(r_preds)

        return (jt.mean(nn.relu(1 + r_f_diff)) + jt.mean(nn.relu(1 - f_r_diff)))


class LogisticGAN(GANLoss):
    def __init__(self, dis):
        super().__init__(dis)

    # gradient penalty
    def R1Penalty(self, real_img, height, alpha):

        real_img = nn.init.constant(real_img.shape, 'float32', real_img)
        assert not real_img.is_stop_grad()
        real_logit = self.dis(real_img, height, alpha)
       
        real_grads = jt.grad(real_logit, real_img).view(real_img.size(0), -1)
 
        r1_penalty = jt.sum(jt.multiply(real_grads, real_grads))
        return r1_penalty

    def dis_loss(self, real_samps, fake_samps, height, alpha, r1_gamma=10.0):
        # Obtain predictions
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        loss = jt.mean(nn.softplus(f_preds)) + jt.mean(nn.softplus(-r_preds))

        if r1_gamma != 0.0:
            r1_penalty = self.R1Penalty(real_samps.detach(), height, alpha) * (r1_gamma * 0.5)
            loss += r1_penalty

        return loss

    def gen_loss(self, _, fake_samps, height, alpha):

        f_preds = self.dis(fake_samps, height, alpha)
        return jt.mean(nn.softplus(-f_preds))
