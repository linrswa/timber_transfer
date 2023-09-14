import torch

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl-gl))
    
    return loss*2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean(1-dg)**2
        gen_losses.append(l)
        loss += l
    
    return loss, gen_losses


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