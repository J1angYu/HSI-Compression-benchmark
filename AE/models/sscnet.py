from torch import nn


def sscnet_cr4(src_channels=202):
    return SpectralSignalsCompressorNetwork(src_channels, target_compression_ratio=4)


def sscnet_cr8(src_channels=202):
    return SpectralSignalsCompressorNetwork(src_channels, target_compression_ratio=8)


def sscnet_cr16(src_channels=202):
    return SpectralSignalsCompressorNetwork(src_channels, target_compression_ratio=16)


def sscnet_cr32(src_channels=202):
    return SpectralSignalsCompressorNetwork(src_channels, target_compression_ratio=32)


class SpectralSignalsCompressorNetwork(nn.Module):
    """
    Title:
        HYPERSPECTRAL DATA COMPRESSION USING FULLY CONVOLUTIONAL AUTOENCODER
    Authors:
        La Grassa, Riccardo and Re, Cristina and Cremonese, Gabriele and Gallo, Ignazio
    Paper:
        https://doi.org/10.3390/rs14102472  
    Cite:
        @article{la2022hyperspectral,
            title={Hyperspectral Data Compression Using Fully Convolutional Autoencoder},
            author={La Grassa, Riccardo and Re, Cristina and Cremonese, Gabriele and Gallo, Ignazio},
            journal={Remote Sensing},
            volume={14},
            number={10},
            pages={2472},
            year={2022},
            publisher={MDPI}
        }
    """

    def __init__(self, src_channels=202, target_compression_ratio=4):
        super(SpectralSignalsCompressorNetwork, self).__init__()

        self.src_channels = src_channels

        self.spatial_downsamplings = 3
        self.spatial_downsampling_factor = 2 ** self.spatial_downsamplings

        self.spectral_downsampling_factor_estimated = target_compression_ratio / self.spatial_downsampling_factor ** 2
        self.latent_channels = int(self.src_channels / self.spectral_downsampling_factor_estimated)
        self.spectral_downsampling_factor = self.src_channels / self.latent_channels

        self.compression_ratio = self.spectral_downsampling_factor * self.spatial_downsampling_factor ** 2
        self.bpppc = 32.0 / self.compression_ratio

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=src_channels,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.PReLU(num_parameters=256),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.PReLU(num_parameters=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.PReLU(num_parameters=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                padding=1,
            ),
            nn.PReLU(num_parameters=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=512,
                out_channels=self.latent_channels,
                kernel_size=3,
                padding=1
            ),
            nn.PReLU(num_parameters=self.latent_channels),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.latent_channels,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.PReLU(num_parameters=512),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=2,
                stride=2,
            ),
            nn.PReLU(num_parameters=256),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=2,
                stride=2,
            ),
            nn.PReLU(num_parameters=256),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=2,
                stride=2,
            ),
            nn.PReLU(num_parameters=256),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=src_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.compress(x)
        x_hat = self.decompress(y)
        return x_hat

    def compress(self, x):
        y = self.encoder(x)
        return y

    def decompress(self, y):
        x_hat = self.decoder(y)
        return x_hat

    @classmethod
    def from_state_dict(cls, state_dict):
        net = cls()
        net.load_state_dict(state_dict)
        return net


if __name__ == '__main__':
    import torchsummary
    model = SpectralSignalsCompressorNetwork()
    print(model)
    torchsummary.summary(model, input_size=(202, 128, 128), batch_size=2, device='cpu')
