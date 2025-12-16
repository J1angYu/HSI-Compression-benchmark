import math
import torch.nn.functional as f

from torch import nn


class ConvolutionalAutoencoder1D(nn.Module):
    """
    Title:
        1D-CONVOLUTIONAL AUTOENCODER BASED HYPERSPECTRAL DATA COMPRESSION
    Authors:
        Kuester, Jannick and Gross, Wolfgang and Middelmann, Wolfgang
    Paper:
        https://doi.org/10.5194/isprs-archives-XLIII-B1-2021-15-2021
    Cite:
        @article{kuester20211d,
            title={1D-convolutional autoencoder based hyperspectral data compression},
            author={Kuester, Jannick and Gross, Wolfgang and Middelmann, Wolfgang},
            journal={International Archives of Photogrammetry, Remote Sensing and Spatial Information Sciences},
            volume={43},
            pages={15--21},
            year={2021},
            publisher={Copernicus GmbH}
        }
    """

    def __init__(self, src_channels=103):
        super(ConvolutionalAutoencoder1D, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=64,
                kernel_size=11,
                stride=1,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=64,
                out_channels=32,
                kernel_size=11,
                stride=1,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=32,
                out_channels=16,
                kernel_size=9,
                stride=1,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=16,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding="same",
            ),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=9,
                stride=1,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Upsample(
                scale_factor=2
            ),
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=11,
                stride=1,
                padding="same",
            ),
            nn.LeakyReLU(),
            nn.Upsample(
                scale_factor=2
            ),
            nn.Conv1d(
                in_channels=64,
                out_channels=1,
                kernel_size=11,
                stride=1,
                padding="same",
            ),
            nn.Sigmoid(),
        )

        self.src_channels = src_channels

        self.spectral_downsamplings = 2
        self.spectral_downsampling_factor_estimated = 2 ** self.spectral_downsamplings

        self.spatial_downsamplings = 0
        self.spatial_downsampling_factor = 2 ** self.spatial_downsamplings

        self.latent_channels = int(math.ceil(self.src_channels / 2 ** self.spectral_downsamplings))
        self.spectral_downsampling_factor = self.src_channels / self.latent_channels
        self.compression_ratio = self.spectral_downsampling_factor * self.spatial_downsampling_factor ** 2
        self.bpppc = 32.0 / self.compression_ratio

        self.padding_amount = 0 if self.src_channels % self.spectral_downsampling_factor_estimated == 0 \
            else self.spectral_downsampling_factor_estimated - self.src_channels % self.spectral_downsampling_factor_estimated

    def compress(self, x):
        if self.padding_amount > 0:
            x = f.pad(x, (self.padding_amount, 0))
        x = x.unsqueeze(1)
        
        y = self.encoder(x)
        y = y.squeeze(1)

        return y

    def decompress(self, y):
        y = y.unsqueeze(1)
        x_hat = self.decoder(y)

        if self.padding_amount > 0:
            x_hat = x_hat[:, :, self.padding_amount:]
        x_hat = x_hat.squeeze(1)

        return x_hat

    def forward(self, x):
        y = self.compress(x)
        x_hat = self.decompress(y)
        return x_hat

    @classmethod
    def from_state_dict(cls, state_dict):
        net = cls()
        net.load_state_dict(state_dict)
        return net


if __name__ == '__main__':
    import torchsummary
    model = ConvolutionalAutoencoder1D()
    print(model)
    torchsummary.summary(model, input_size=(103), batch_size=2, device='cpu')