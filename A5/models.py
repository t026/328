import torch
import math

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import List
from typing import Tuple
from einops import rearrange #pip install einops




class VarianceScheduler:
    def __init__(self, beta_start: int=0.0001, beta_end: int=0.02, num_steps: int=1000, interpolation: str='linear') -> None:
        self.num_steps = num_steps

        # find the beta valuess by linearly interpolating from start beta to end beta
        if interpolation == 'linear':
            # TODO: complete the linear interpolation of betas here
            self.betas = torch.linspace(beta_start, beta_end, self.num_steps)
        elif interpolation == 'quadratic':
            # TODO: complete the quadratic interpolation of betas here
            self.betas = torch.square(torch.linspace(math.sqrt(beta_start), math.sqrt(beta_end), self.num_steps))
        else:
            raise Exception('[!] Error: invalid beta interpolation encountered...')
        

        # TODO: add other statistics such alphas alpha_bars and all the other things you might need here
        self.alpha = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alpha, 0)
        
        

    def add_noise(self, x:torch.Tensor, time_step:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        # Ensure time_step is shaped properly

        # TODO: sample a random noise
        noise = torch.randn_like(x)

        # TODO: construct the noisy sample
        a_t = self.alpha_bars.to(device)[time_step]
        a_t = a_t.view(-1, 1, 1, 1)
        noisy_input = torch.sqrt(a_t) * x + torch.sqrt(1 - a_t) * noise
        return noisy_input, noise


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
      super().__init__()

      self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # TODO: compute the sinusoidal positional encoding of the time
        device = time.device


        emb = torch.arange((self.dim // 2), device=device).float()
        emb = torch.pow(10000, -emb/(self.dim // 2))  
        time = time.unsqueeze(1) 
        embeddings = time * emb  
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)  
        return embeddings


#Reference for Unet - https://github.com/randomaccess2023/MG2023/tree/main/Video%2054
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels = None, residual=False):
        super(ResBlock, self).__init__()
        
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.resnet_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
                                        nn.GroupNorm(8, mid_channels),
                                        nn.SiLU(),
                                        nn.Conv2d(mid_channels, out_channels, 3, 1, 1),
                                        nn.GroupNorm(8, out_channels))
        
    def forward(self, x):
        if self.residual:
            return x + self.resnet_conv(x)
        else:
            return self.resnet_conv(x)
        
class SelfAttentionBlock(nn.Module):
    def __init__(self, num_channels):
        super(SelfAttentionBlock, self).__init__()
        
        self.attn_norm = nn.GroupNorm(8, num_channels)
        self.mha = nn.MultiheadAttention(num_channels, 4, True)
        
    def forward(self, x):
        b, c, h, w = x.shape
        inp_attn = x.reshape(b, c, h*w)
        inp_attn = self.attn_norm(inp_attn).transpose(1,2)
        out_attn, _ = self.mha(inp_attn, inp_attn, inp_attn)
        out_attn = out_attn.transpose(1, 2).reshape(b, c, h, w)
        return x + out_attn

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=128):
        super(DownBlock, self).__init__()
        
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(in_channels, in_channels, residual=True),
            ResBlock(in_channels, out_channels)
        )
        
        self.time_emb_layers = nn.Sequential(nn.SiLU(),
                                          nn.Linear(time_emb_dim, out_channels))
        
    def forward(self, x, t):
        x = self.down(x)
        time_emb = self.time_emb_layers(t)[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])
        return x + time_emb

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=128):
        super(UpBlock, self).__init__()
        
        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up = nn.Sequential(
            ResBlock(in_channels, in_channels, residual=True),
            ResBlock(in_channels, out_channels, in_channels//2)
        )
        
        self.time_emb_layers = nn.Sequential(nn.SiLU(), 
                                          nn.Linear(time_emb_dim, out_channels))
        
    def forward(self, x, skip, t):
        x = self.upsamp(x)
        x = torch.cat([skip, x], dim=1)
        x = self.up(x)
        time_emb = self.time_emb_layers(t)[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])
        return x + time_emb

class UNet(nn.Module):
    def __init__(self, in_channels: int=1, 
                 down_channels: List=[64, 128, 128, 128, 128], 
                 up_channels: List=[128, 128, 128, 128, 64], 
                 time_emb_dim: int=128,
                 num_classes: int=10) -> None:
        super().__init__()

        # NOTE: You can change the arguments received by the UNet if you want, but keep the num_classes argument

        self.num_classes = num_classes

        # TODO: time embedding layer
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # TODO: define the embedding layer to compute embeddings for the labels
        self.class_emb = nn.Embedding(self.num_classes, time_emb_dim)

        # define your network architecture here
        self.r1 = ResBlock(in_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.a1 = SelfAttentionBlock(128)
        self.down2 = DownBlock(128, 256)
        self.a2 = SelfAttentionBlock(256)
        self.down3 = DownBlock(256, 256)
        self.a3 = SelfAttentionBlock(256)
        
        self.lat = nn.Sequential(ResBlock(256, 512),
                                ResBlock(512, 512),
                                ResBlock(512, 256))

        self.up1 = UpBlock(512, 128)
        self.a4 = SelfAttentionBlock(128)
        self.up2 = UpBlock(256, 64)
        self.a5 = SelfAttentionBlock(64)
        self.up3 = UpBlock(128, 64)
        self.a6 = SelfAttentionBlock(64)
        
        self.out = nn.Conv2d(64, 1, 1)        
    def forward(self, x: torch.Tensor, timestep: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
        # TODO: embed time
        t = self.time_mlp(timestep)

        # TODO: handle label embeddings if labels are avaialble
        if label is not None:
            l = self.class_emb(label)
            t += l
        
        # TODO: compute the output of your network
        
        x1 = self.r1(x)
        x2 = self.down1(x1, t)
        x2 = self.a1(x2)
        x3 = self.down2(x2, t)
        x3 = self.a2(x3)
        x4 = self.down3(x3, t)
        x4 = self.a3(x4)
        
        x4 = self.lat(x4)
        
        x = self.up1(x4, x3, t)
        x = self.a4(x)
        x = self.up2(x, x2, t)
        x = self.a5(x)
        x = self.up3(x, x1, t)
        x = self.a6(x)

        out = self.out(x)

        return out




class VAE(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 height: int=32, 
                 width: int=32, 
                 mid_channels: List=[32, 32, 32], 
                 latent_dim: int=32, 
                 num_classes: int=10) -> None:
        
        super().__init__()

        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # NOTE: self.mid_size specifies the size of the image [C, H, W] in the bottleneck of the network
        self.mid_size = [mid_channels[-1], height // (2 ** (len(mid_channels)-1)), width // (2 ** (len(mid_channels)-1))]  #32, 128, 128

        # NOTE: You can change the arguments of the VAE as you please, but always define self.latent_dim, self.num_classes, self.mid_size
        
        # TODO: handle the label embedding here
        self.class_emb = nn.Sequential(
            nn.Linear(num_classes, 4*4*128),
            nn.ReLU(),
        )

        
        # TODO: define the encoder part of your network
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(1)
        )

        # TODO: define the network/layer for estimating the mean
        self.mean_net = nn.Linear(4*4*128, latent_dim)
        
        # TODO: define the networklayer for estimating the log variance
        self.logvar_net = nn.Linear(4*4*128, latent_dim)

        # TODO: define the decoder part of your network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 4*4*128, 128*8*8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # TODO: compute the output of the network encoder
        out = self.encoder(x)

        # TODO: estimating mean and logvar
        mean = self.mean_net(out)
        logvar = self.logvar_net(out)
        
        # TODO: computing a sample from the latent distribution
        sample = self.reparameterize(mean, logvar)

        # TODO: decoding the sample
        out = self.decode(sample, label)
        return out, mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: implement the reparameterization trick: sample = noise * std + mean
        std = torch.exp(0.5 * logvar)
        sample = mean + std * torch.randn_like(std)

        return sample
    
    @staticmethod
    def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # TODO: compute the binary cross entropy between the pred (reconstructed image) and the traget (ground truth image)
        loss = F.binary_cross_entropy(pred, target, reduction='sum')

        return loss
       
    @staticmethod
    def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: compute the KL divergence
        kl_div = -.5 * (logvar.flatten(start_dim=1) + 1 - torch.exp(logvar.flatten(start_dim=1)) - mean.flatten(start_dim=1).pow(2)).sum()

        return kl_div

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device=torch.device('cpu'), labels: torch.Tensor=None):
        if labels is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        else:
            # randomly consider some labels
            labels = torch.randint(0, self.num_classes, [num_samples,], device=device)

        # TODO: sample from standard Normal distrubution
        noise = torch.randn(num_samples, self.latent_dim, device=device)


        # TODO: decode the noise based on the given labels
        out = self.decode(noise, labels)


        return out
    
    def decode(self, sample: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # TODO: use you decoder to decode a given sample and their corresponding labels
        label_one_hot = F.one_hot(labels, self.num_classes).float()
        label_embedding = self.class_emb(label_one_hot)

        sample = torch.cat([sample, label_embedding], dim=1)
        

        # Decode
        out = self.decoder(sample)



        return out


class LDDPM(nn.Module):
    def __init__(self, network: nn.Module, vae: VAE, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.vae = vae
        self.network = network

        # freeze vae
        self.vae.requires_grad_(False)
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # TODO: uniformly sample as many timesteps as the batch size
        t = ...

        # TODO: generate the noisy input
        noisy_input, noise = ...

        # TODO: estimate the noise
        estimated_noise = ...

        # compute the loss (either L1 or L2 loss)
        loss = F.mse_loss(estimated_noise, noise)

        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # TODO: implement the sample recovery strategy of the DDPM
        sample = ...

        return sample

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cpu'), labels: torch.Tensor=None):
        if labels is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        else:
            labels = torch.randint(0, self.vae.num_classes, [num_samples,], device=device)
        
        # TODO: using the diffusion model generate a sample inside the latent space of the vae
        # NOTE: you need to recover the dimensions of the image in the latent space of your VAE
        sample = ...

        sample = self.vae.decode(sample, labels)
        
        return sample


class DDPM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.network = network
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: uniformly sample as many timesteps as the batch size
        t = torch.randint(0, self.var_scheduler.num_steps, (x.size(0),), device = x.device)
        # TODO: generate the noisy input
        noisy_input, noise = self.var_scheduler.add_noise(x, t)

        # TODO: estimate the noise
        estimated_noise = self.network(noisy_input, t, label)
        # TODO: compute the loss (either L1, or L2 loss)
        loss = F.l1_loss(estimated_noise, noise)
        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # TODO: implement the sample recovery strategy of the DDPM
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        timestep = timestep.to(device)
        sample = None
        alpha = self.var_scheduler.alpha.to(device)
        alpha_hat = self.var_scheduler.alpha_bars.to(device)
        beta = self.var_scheduler.betas.to(device)
        # Ensure alpha, alpha_hat, and beta are broadcastable to the shape of the noisy sample
        alpha_t = alpha[timestep].view(-1, 1, 1, 1)  # Shape: [num_samples, 1, 1, 1]
        alpha_hat_t = alpha_hat[timestep].view(-1, 1, 1, 1)  # Shape: [num_samples, 1, 1, 1]
        beta_t = beta[timestep].view(-1, 1, 1, 1)  # Shape: [num_samples, 1, 1, 1]
        # Broadcast these to match the sample shape [num_samples, 1, 32, 32]
        mean = 1 / torch.sqrt(alpha_t) * (noisy_sample - (beta_t / torch.sqrt(1 - alpha_hat_t)) * estimated_noise)
        if torch.min(timestep) > 0:
            alp = alpha_hat[timestep-1].view(-1, 1, 1, 1)
            sigma_square = ((1-alp)/(1-alpha_hat_t)) * beta_t
            sample = mean + torch.sqrt(sigma_square) *torch.randn_like(mean)
        else:
            sample = mean

        return sample

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cpu'), labels: torch.Tensor=None):
        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, [num_samples,], device=device)
        else:
            labels = None

        # TODO: apply the iterative sample generation of the DDPM
        sample = torch.randn(num_samples, 1, 32, 32, device=device)

        timesteps = torch.arange(self.var_scheduler.num_steps - 1, -1, -1, device=device)  #reverse order

        for t in timesteps:
            # Create timestep tensor
            timestep = torch.full((num_samples,), t.item(), device=device, dtype=torch.long)
            
            # Estimate the noise for the entire batch
            estimated_noise = self.network(sample, timestep, labels)
            
            # Recover the sample for the entire batch
            sample = self.recover_sample(sample, estimated_noise, timestep)

                
        return sample


class DDIM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.network = network
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: uniformly sample as many timesteps as the batch size
        t = torch.randint(0, self.var_scheduler.num_steps, (x.size(0),), device = x.device)
        # TODO: generate the noisy input
        noisy_input, noise = self.var_scheduler.add_noise(x, t)

        # TODO: estimate the noise
        estimated_noise = self.network(noisy_input, t, label)
        # TODO: compute the loss (either L1, or L2 loss)
        loss = F.l1_loss(estimated_noise, noise)
        return loss

    
    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # TODO: apply the sample recovery strategy of the DDIM

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        timestep = timestep.to(device)
        alpha_hat = self.var_scheduler.alpha_bars.to(device)
        alpha_hat_t = alpha_hat[timestep].view(-1, 1, 1, 1) 
        beta = self.var_scheduler.betas.to(device)
        alpha_hat_minus = alpha_hat[timestep-1].view(-1, 1, 1, 1)
        beta_t = beta[timestep].view(-1, 1, 1, 1)  # Shape: [num_samples, 1, 1, 1]
        sigma = torch.sqrt(((1-alpha_hat_minus)/(1-alpha_hat_t)) * beta_t)
        
        prediction = torch.sqrt(alpha_hat_minus) * ((noisy_sample-torch.sqrt(1-alpha_hat_t)*estimated_noise)/torch.sqrt(alpha_hat_t))
        direction = torch.sqrt(1- alpha_hat_minus)*estimated_noise
        sample = prediction + direction + sigma * torch.randn_like(noisy_sample)
        
        return sample
    
    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cpu'), labels: torch.Tensor=None):
        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, [num_samples,], device=device)
        else:
            labels = None
        # TODO: apply the iterative sample generation of DDIM (similar to DDPM)
        sample = torch.randn(num_samples, 1, 32, 32, device=device)

        timesteps = torch.arange(self.var_scheduler.num_steps - 1, -1, -1, device=device)  #reverse order

        for t in timesteps:
            # Create timestep tensor
            timestep = torch.full((num_samples,), t.item(), device=device, dtype=torch.long)
            
            # Estimate the noise for the entire batch
            estimated_noise = self.network(sample, timestep, labels)
            
            # Recover the sample for the entire batch
            sample = self.recover_sample(sample, estimated_noise, timestep)

                
        return sample

