from torch import nn
import torch.nn.functional as f


def cae3d_cr4(src_channels=202):
    return ConvolutionalAutoencoder3D(src_channels=src_channels, latent_channels=64)


def cae3d_cr8(src_channels=202):
    return ConvolutionalAutoencoder3D(src_channels=src_channels, latent_channels=32)


def cae3d_cr16(src_channels=202):
    return ConvolutionalAutoencoder3D(src_channels=src_channels, latent_channels=16)


def cae3d_cr32(src_channels=202):
    return ConvolutionalAutoencoder3D(src_channels=src_channels, latent_channels=8)


class ConvolutionalAutoencoder3D(nn.Module):
    """
    Title:
        END-TO-END JOINT SPECTRAL-SPATIAL COMPRESSION AND RECONSTRUCTION OF HYPERSPECTRAL IMAGES USING A 3D CONVOLUTIONAL AUTOENCODER
    Authors:
        Chong, Yanwen and Chen, Linwei and Pan, Shaoming
    Paper:
        https://doi.org/10.1117/1.JEI.30.4.041403
    Cite:
        @article{chong2021end,
            title={End-to-end joint spectral--spatial compression and reconstruction of hyperspectral images using a 3D convolutional autoencoder},
            author={Chong, Yanwen and Chen, Linwei and Pan, Shaoming},
            journal={Journal of Electronic Imaging},
            volume={30},
            number={4},
            pages={041403},
            year={2021},
            publisher={SPIE}
        }
    """

    def __init__(self, src_channels=202, latent_channels=16):
        super(ConvolutionalAutoencoder3D, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=16,
                kernel_size=(2, 5, 5),
                stride=(1, 2, 2),
                padding=(0, 2, 2),
            ),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=(5, 5, 5),
                stride=(2, 2, 2),
                padding=(2, 2, 2),
            ),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            _ResBlock(32),
            _ResBlock(32),
            _ResBlock(32),
            nn.Conv3d(
                in_channels=32,
                out_channels=latent_channels,
                kernel_size=(5, 5, 5),
                stride=(2, 2, 2),
                padding=(2, 2, 2),
            ),
            nn.BatchNorm3d(latent_channels),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=(2, 4, 4)),
            nn.Conv3d(
                in_channels=latent_channels,
                out_channels=32,
                kernel_size=(5, 5, 5),
                stride=(2, 2, 2),
                padding=(2, 2, 2),
            ),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            _ResBlock(32),
            _ResBlock(32),
            _ResBlock(32),
            nn.Upsample(scale_factor=(4, 2, 2)),
            nn.Conv3d(
                in_channels=32,
                out_channels=16,
                kernel_size=(5, 3, 3),
                stride=(2, 1, 1),
                padding=(2, 1, 1),
            ),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=(4, 2, 2)),
            nn.Conv3d(
                in_channels=16,
                out_channels=1,
                kernel_size=(2, 3, 3),
                stride=(2, 1, 1),
                padding=(0, 1, 1),  # (2, 1, 1)
            ),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(),
        )

        self.src_channels = src_channels
        self.latent_channels = latent_channels


        self.spectral_downsamplings = 2
        self.spectral_downsampling_factor_estimated = 2 ** self.spectral_downsamplings

        self.padding_amount = 0 if self.src_channels % self.spectral_downsampling_factor_estimated == 0 \
            else self.spectral_downsampling_factor_estimated - self.src_channels % self.spectral_downsampling_factor_estimated
        
        self.spectral_downsampling_factor = self.src_channels / ((self.src_channels + self.padding_amount) / self.spectral_downsampling_factor_estimated)

        self.spatial_downsamplings = 3
        self.spatial_downsampling_factor = 2 ** self.spatial_downsamplings

        self.compression_ratio = self.spectral_downsampling_factor * self.spatial_downsampling_factor ** 2 / self.latent_channels
        self.bpppc = 32.0 / self.compression_ratio


    def forward(self, x):
        if self.padding_amount > 0:
            x = f.pad(x, (0, 0, 0, 0, self.padding_amount, 0))
        x = x.unsqueeze(1)

        y = self.encoder(x)
        x_hat = self.decoder(y)

        if self.padding_amount > 0:
            x_hat = x_hat[:, :, self.padding_amount:]
        x_hat = x_hat.squeeze(1)

        return x_hat

    def compress(self, x):
        if self.padding_amount > 0:
            x = f.pad(x, (0, 0, 0, 0, self.padding_amount, 0))
        x = x.unsqueeze(1)
        y = self.encoder(x)
        y = y.squeeze(1)
        return y

    def decompress(self, y):
        x_hat = self.decoder(y)

        if self.padding_amount > 0:
            x_hat = x_hat[:, :, self.padding_amount:]

        x_hat = x_hat.squeeze(1)
        return x_hat

    @classmethod
    def from_state_dict(cls, state_dict):
        net = cls()
        net.load_state_dict(state_dict)
        return net


class _ResBlock(nn.Module):
    def __init__(self, channels):
        super(_ResBlock, self).__init__()

        self.act = nn.LeakyReLU()

        self.block = nn.Sequential(*[
            nn.Conv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(channels),
            self.act,
            nn.Conv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(channels),
        ])

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.act(out)
        return out


if __name__ == '__main__':
    import torchsummary
    model = ConvolutionalAutoencoder3D()
    print(model)
    torchsummary.summary(model, input_size=(202, 128, 128), batch_size=2, device='cpu')
