import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image


# Beta schedule generator
def get_beta_schedule(schedule, timesteps):
    if schedule == "linear":
        return torch.linspace(1e-4, 2e-2, timesteps)
    elif schedule == "cosine":
        s = 0.008
        steps = torch.arange(timesteps + 1, dtype=torch.float64)
        alphas_cumprod = torch.cos(
            ((steps / timesteps) + s) / (1 + s) * (torch.pi / 2)
        ) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return torch.clip(betas, 0, 0.999)
    else:
        raise ValueError("Unknown beta schedule!")

# Noise sampling function
def q_sample(x0, t, betas):
    noise = torch.randn_like(x0)
    alpha_cumprod = torch.cumprod(1 - betas, dim=0)
    sqrt_alpha_cumprod = alpha_cumprod[t].sqrt()
    sqrt_one_minus_alpha_cumprod = (1 - alpha_cumprod[t]).sqrt()
    return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise, noise


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, padding=1),
            nn.GroupNorm(1, dim_out),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
            nn.GroupNorm(1, dim_out),
        )
        self.residual = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.residual(x)

class UNetWithAttention(nn.Module):
    def __init__(self, dim, channels=3, time_embedding_dim=128):
        super().__init__()
        self.channels = channels
        self.time_embedding_dim = time_embedding_dim

        # Sinusoidal positional embedding for time
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embedding_dim),  # Time input is scalar
            nn.GELU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        # UNet backbone
<<<<<<< Updated upstream
        self.down1 = ResBlock(channels, dim)
        self.down2 = ResBlock(dim, dim * 2)
        self.mid = ResBlock(dim * 2, dim * 2)
        self.up1 = ResBlock(dim * 2, dim)
        self.up2 = ResBlock(dim, channels)
=======
        self.down1 = nn.Conv2d(channels, dim, 3, padding=1)
        self.down2 = nn.Conv2d(dim, dim * 2, 3, padding=1)
        self.mid = nn.Conv2d(dim * 2, dim * 2, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(dim * 2, dim, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(dim, channels, 3, padding=1)
>>>>>>> Stashed changes

    def forward(self, x, t):
        # Time embedding
        time_emb = self.time_mlp(t.float().unsqueeze(1))  # Fix: Convert t to float
<<<<<<< Updated upstream
        # Shape: [B, time_embedding_dim]
=======
 # Shape: [B, time_embedding_dim]
>>>>>>> Stashed changes

        # Downsample
        h1 = self.down1(x)
        h2 = self.down2(h1)

        # Bottleneck (use time embedding here if needed for advanced models)
        h3 = self.mid(h2)

        # Upsample
        h4 = self.up1(h3 + h2)  # Skip connection
        h5 = self.up2(h4 + h1)  # Skip connection

        return h5


class DiffusionProcess(nn.Module):
    def __init__(self, model, beta_scheduler="cosine", timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.betas = get_beta_schedule(beta_scheduler, timesteps)
        self.alphas_cumprod = torch.cumprod(1 - self.betas, dim=0)

    def forward(self, x, t):
        return self.model(x)

    def sample(self, shape, device="cuda"):
        x = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(self.timesteps)), desc="Sampling"):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred_noise = self.model(x, t_tensor)
            alpha_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            if t > 0:
                noise = torch.randn_like(x)
                x = (1 / (1 - beta_t).sqrt()) * (x - beta_t / (1 - alpha_t).sqrt() * pred_noise) + noise
            else:
                x = (1 / (1 - beta_t).sqrt()) * (x - beta_t / (1 - alpha_t).sqrt() * pred_noise)
        return x
    
device = "cuda" if torch.cuda.is_available() else "cpu"

# initialize the diffusion model
unet = UNetWithAttention(dim=64, channels=3).to(device)
diffusion_model = DiffusionProcess(unet, beta_scheduler="cosine", timesteps=1000).to(device)

batch_size = 1
image_size = 32
channels = 3
generated_image = diffusion_model.sample(
    shape=(batch_size, channels, image_size, image_size), device=device
)

<<<<<<< Updated upstream
save_image((generated_image + 1) / 2, "generated_image5.png")  # Scale [-1, 1] to [0, 1]
print("Image generated and saved as 'generated_image5.png'")
=======
save_image((generated_image + 1) / 2, "generated_image4.png")  # Scale [-1, 1] to [0, 1]
print("Image generated and saved as 'generated_image4.png'")
>>>>>>> Stashed changes
