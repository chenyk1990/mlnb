# DDPM-min.py
import math, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# -----------------------
# 1) Hyperparams
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
img_size = 28
batch_size = 128
T = 1000                      # diffusion steps
lr = 2e-4
epochs = 10                   # bump to 5–10 for nicer samples

# -----------------------
# 2) Beta schedule + precompute
# -----------------------
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

betas = linear_beta_schedule(T).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), value=1.0)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# -----------------------
# 3) Forward diffusion q(x_t | x_0)
# -----------------------
def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    s1 = extract(sqrt_alphas_cumprod, t, x0.shape)
    s2 = extract(sqrt_one_minus_alphas_cumprod, t, x0.shape)
    return s1 * x0 + s2 * noise

def extract(a, t, x_shape):
    """gather coefficients at batch indices t and reshape"""
    out = a.gather(-1, t).float()
    return out.view(-1, *([1] * (len(x_shape) - 1)))

# -----------------------
# 4) Tiny UNet-ish epsilon-predictor
# -----------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        # t in [0, T-1]
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(math.log(1e-4), math.log(1e4), half, device=device))
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(time_dim, out_ch)
        self.act = nn.SiLU()
    def forward(self, x, temb):
        h = self.act(self.conv1(x))
        h = h + self.time(temb)[:, :, None, None]
        h = self.act(h)
        h = self.conv2(h)
        return self.act(h)

class TinyUNet(nn.Module):
    def __init__(self, img_ch=1, base=64, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU(),
        )
        self.down1 = Block(img_ch, base, time_dim)
        self.down2 = Block(base, base*2, time_dim)
        self.pool = nn.AvgPool2d(2)
        self.up1 = Block(base*2, base, time_dim)
        self.out = nn.Conv2d(base, img_ch, 3, padding=1)
    def forward(self, x, t):
        temb = self.time_mlp(t)
        d1 = self.down1(x, temb)
        d2 = self.down2(self.pool(d1), temb)
        u1 = F.interpolate(d2, scale_factor=2, mode='nearest')
        u1 = self.up1(u1, temb)
        h = u1 + d1                       # tiny skip
        return self.out(h)

model = TinyUNet().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr)

# -----------------------
# 5) Data
# -----------------------
tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*2-1)   # scale to [-1,1]
])
trainset = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

# -----------------------
# 6) Loss (epsilon-prediction)
# -----------------------
def loss_fn(x0):
    b = x0.size(0)
    t = torch.randint(0, T, (b,), device=device).long()
    noise = torch.randn_like(x0)
    xt = q_sample(x0, t, noise)
    eps_pred = model(xt, t)
    return F.mse_loss(eps_pred, noise)

# -----------------------
# 7) Training
# -----------------------
model.train()
for epoch in range(epochs):
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
    for x, _ in pbar:
        x = x.to(device)
        opt.zero_grad()
        loss = loss_fn(x)
        loss.backward()
        opt.step()
        pbar.set_postfix(loss=float(loss))

# -----------------------
# 8) Sampling (p_theta)
# -----------------------
@torch.no_grad()
def p_sample(x, t):
    bet = extract(betas, t, x.shape)
    sqrt_recip_a = extract(sqrt_recip_alphas, t, x.shape)
    s1mac = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)

    eps_theta = model(x, t)
    mean = sqrt_recip_a * (x - bet / s1mac * eps_theta)
    if (t == 0).all():
        return mean
    var = extract(posterior_variance, t, x.shape)
    noise = torch.randn_like(x)
    return mean + torch.sqrt(var) * noise

@torch.no_grad()
def sample(n=16):
    model.eval()
    x = torch.randn(n, 1, img_size, img_size, device=device)
    for t in reversed(range(T)):
        x = p_sample(x, torch.full((n,), t, device=device, dtype=torch.long))
    return (x.clamp(-1,1) + 1) / 2  # back to [0,1]

# -----------------------
# 9) Save a grid of samples
# -----------------------
from torchvision.utils import make_grid, save_image
samples = sample(36)
grid = make_grid(samples, nrow=6)
save_image(grid, "ddpm_mnist_samples_10.png")
# save_image(grid, "ddpm_mnist_samples.png")
print("Saved samples to ddpm_mnist_samples.png")
