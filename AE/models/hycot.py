import math
import torch
import torch.nn.functional as f
from torch import nn

from einops import rearrange, repeat


def hycot_cr4(src_channels=202):
    return HyperspectralCompressionTransformer(
        src_channels=src_channels,
        target_compression_ratio=4,
    )


def hycot_cr8(src_channels=202):
    return HyperspectralCompressionTransformer(
        src_channels=src_channels,
        target_compression_ratio=8,
    )


def hycot_cr16(src_channels=202):
    return HyperspectralCompressionTransformer(
        src_channels=src_channels,
        target_compression_ratio=16,
    )


def hycot_cr32(src_channels=202):
    return HyperspectralCompressionTransformer(
        src_channels=src_channels,
        target_compression_ratio=32,
    )


class HyperspectralCompressionTransformer(nn.Module):
    def __init__(
            self,
            src_channels=202,
            target_compression_ratio=4,
            patch_depth=4,
            hidden_dim=1024,
            dim=64,
            depth=5,
            heads=4,
            mlp_dim=8,
            dim_head=16,
            dropout=0.,
            emb_dropout=0.,
        ):
        super().__init__()

        self.src_channels = src_channels

        self.dim = dim

        latent_channels = int(math.ceil(src_channels / target_compression_ratio))
        self.latent_channels = latent_channels

        self.compression_ratio = src_channels / latent_channels
        self.bpppc = 32 / self.compression_ratio

        self.delta_pad = int(math.ceil(src_channels / patch_depth)) * patch_depth - src_channels

        num_patches = (src_channels + self.delta_pad) // patch_depth
        self.num_patches = num_patches

        patch_dim = (src_channels + self.delta_pad) // num_patches
        self.patch_dim = patch_dim
        
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.comp_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, 'ViT')

        self.to_latent = nn.Sequential(
            nn.Linear(
                in_features=dim,
                out_features=hidden_dim,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=hidden_dim,
                out_features=latent_channels,
            ),
            nn.Sigmoid(),
        )

        self.patch_deembed = nn.Sequential(
            nn.Linear(
                in_features=latent_channels,
                out_features=hidden_dim,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=hidden_dim,
                out_features=src_channels,
            ),
            nn.Sigmoid(),
        )

    def compress(self, x):
        _, _, h, w = x.shape

        if self.delta_pad > 0:
            x = f.pad(x, (0, 0, 0, 0, self.delta_pad, 0))

        x = rearrange(x, 'b (n pd) w h -> (b w h) n pd',
                      n = self.num_patches,
                      pd = self.patch_dim,
                      )

        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        # concat compression tokens
        comp_tokens = repeat(self.comp_token, '() n d -> b n d', b = b)
        x = torch.cat((comp_tokens, x), dim = 1)

        # add position embedding
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.transformer(x)

        # extract transformed comp_tokens
        y = x[:, 0]
        
        y = self.to_latent(y)

        y = rearrange(y, '(b w h) d -> b d w h',
                      d = self.latent_channels,
                      w = w,
                      h = h,
                      )

        return y

    def decompress(self, y):
        y = rearrange(y, 'b d w h -> b w h d')
        
        x_hat = self.patch_deembed(y)
        
        x_hat = rearrange(x_hat, 'b w h c -> b c w h')

        return x_hat
    
    def forward(self, x):
        y = self.compress(x)
        x_hat = self.decompress(y)
        return x_hat


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = f.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout = dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth-2):
            self.skipcat.append(nn.Conv2d(num_channel+1, num_channel+1, [1, 2], 1, 0))

    def forward(self, x, mask = None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x, mask = mask)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:           
                last_output.append(x)
                if nl > 1:             
                    x = self.skipcat[nl-2](torch.cat([x.unsqueeze(3), last_output[nl-2].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x, mask = mask)
                x = ff(x)
                nl += 1

        return x


if __name__ == '__main__':
    import torchsummary
    model = HyperspectralCompressionTransformer()
    print(model)
    torchsummary.summary(model, input_size=(202, 128, 128), batch_size=2, device='cpu')
