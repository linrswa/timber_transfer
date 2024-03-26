import torch

from .utils import multiscale_fft, safe_log

class LossCollector:
    
    @staticmethod
    def multiscale_fft_loss(s, y_g_hat, reduction='mean'):
        ori_stft = multiscale_fft(s.squeeze(dim=1))
        rec_stft = multiscale_fft(y_g_hat.squeeze(dim=1))
        loss_gen_multiscale_fft = 0 
        reduction_fn = torch.sum if reduction == 'sum' else torch.mean
        for s_x, s_y in zip(ori_stft, rec_stft):
            linear_loss = reduction_fn((s_x - s_y).abs())
            log_loss = reduction_fn((safe_log(s_x) - safe_log(s_y)).abs())
            loss_gen_multiscale_fft += linear_loss + log_loss
        return loss_gen_multiscale_fft

    @staticmethod    
    def feature_loss(fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl-gl))
        return loss*2

    @staticmethod    
    def discriminator_loss(disc_real_outputs, disc_generated_outputs):
        loss = 0
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1-dr)**2)
            g_loss = torch.mean(dg**2)
            loss += (r_loss + g_loss)
        return loss

    @staticmethod    
    def generator_loss(disc_outputs):
        loss = 0
        for dg in disc_outputs:
            l = torch.mean(1-dg)**2
            loss += l
        return loss

    @staticmethod    
    def kl_loss(mu, logvar):
        r"""Returns the Kullback-Leibler divergence loss with a standard Gaussian.

        Args:
            mu (nn.Variable): Mean of the distribution of shape (B, D, 1).
            logvar (nn.Variable): Log variance of the distribution of
                shape (B, D, 1).

        Returns:
            nn.Variable: Kullback-Leibler divergence loss.
        """
        return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar, axis=1))