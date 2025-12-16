import csv
import numpy as np
import os
import torch

from einops import rearrange
from torch.utils.data import Dataset


class HySpecNet11k(Dataset):
    """
    Dataset:
        HySpecNet-11k
    Authors:
        Martin Hermann Paul Fuchs
        Begüm Demir
    Related Paper:
        HySpecNet-11k: A Large-Scale Hyperspectral Dataset for Benchmarking Learning-Based Hyperspectral Image Compression Methods
        https://arxiv.org/abs/2306.00385
    Cite:
        @inproceedings{fuchs2023hyspecnet,
            title={Hyspecnet-11k: A large-scale hyperspectral dataset for benchmarking learning-based hyperspectral image compression methods},
            author={Fuchs, Martin Hermann Paul and Demir, Beg{\"u}m},
            booktitle={IGARSS 2023-2023 IEEE International Geoscience and Remote Sensing Symposium},
            pages={1779--1782},
            year={2023},
            organization={IEEE}
        }

    Folder Structure:
        - root_dir/
            - patches/
                - tile_001/
                    - tile_001-patch_01/
                        - tile_001-patch_01-DATA.npy
                        - tile_001-patch_01-QL_PIXELMASK.TIF
                        - tile_001-patch_01-QL_QUALITY_CIRRUS.TIF
                        - tile_001-patch_01-QL_QUALITY_CLASSES.TIF
                        - tile_001-patch_01-QL_QUALITY_CLOUD.TIF
                        - tile_001-patch_01-QL_QUALITY_CLOUDSHADOW.TIF
                        - tile_001-patch_01-QL_QUALITY_HAZE.TIF
                        - tile_001-patch_01-QL_QUALITY_SNOW.TIF
                        - tile_001-patch_01-QL_QUALITY_TESTFLAGS.TIF
                        - tile_001-patch_01-QL_SWIR.TIF
                        - tile_001-patch_01-QL_VNIR.TIF
                        - tile_001-patch_01-SPECTRAL_IMAGE.TIF
                        - tile_001-patch_01-THUMBNAIL.jpg
                    - tile_001-patch_02/
                        - ...
                    - ...
                - tile_002/
                    - ...
                - ...
            - splits/
                - easy/
                    - test.csv
                    - train.csv
                    - val.csv
                - hard/
                    - test.csv
                    - train.csv
                    - val.csv
                - ...
            - ...
    """
    def __init__(self, root_dir, mode="easy", split="train", transform=None, random_subsample_factor=None):
        self.root_dir = root_dir

        self.csv_path = os.path.join(self.root_dir, "splits", mode, f"{split}.csv")
        with open(self.csv_path, newline='') as f:
            csv_reader = csv.reader(f)
            csv_data = list(csv_reader)
            self.npy_paths = sum(csv_data, [])
        self.npy_paths = [os.path.join(self.root_dir, "patches", x) for x in self.npy_paths]

        self.transform = transform

        assert random_subsample_factor is None or np.log2(random_subsample_factor) % 1 ==  0
        self.random_subsample_factor = random_subsample_factor

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, index):
        # get full numpy path
        npy_path = self.npy_paths[index]
        # read numpy data
        img = np.load(npy_path)
        # convert numpy array to pytorch tensor
        img = torch.from_numpy(img)
        # apply transformations
        if self.transform:
            img = self.transform(img)
        # pick random pixels:
        if self.random_subsample_factor:
            c, h, w = img.shape

            sample_size = int(h / self.random_subsample_factor) ** 2

            flattened_tensor = img.flatten(1, 2)

            num_elements = flattened_tensor.size(1)

            random_indixes = torch.randperm(num_elements)[:sample_size]

            subsampled_tensor = flattened_tensor[:, random_indixes]

            img = rearrange(subsampled_tensor, 'c (h w) -> c h w',
                h = int(h / self.random_subsample_factor),
                w = int(w / self.random_subsample_factor),
            )
        return img


if __name__ == '__main__':
    import torchvision

    # dataset = HySpecNet11k(
    #     "./datasets/hyspecnet-11k/",
    #     split="train",
    #     mode="easy",
    #     transform=torchvision.transforms.CenterCrop(32),
    # )
    # for data in dataset:
    #     print("length:", len(dataset))

    #     print("shape:", data.shape)
    #     print("dtype:", data.dtype)
    #     break

    dataset = HySpecNet11k(
        "./datasets/hyspecnet-11k/",
        split="train",
        mode="easy",
        random_subsample_factor=8,
    )
    for data in dataset:
        print("length:", len(dataset))

        print("shape:", data.shape)
        print("dtype:", data.dtype)
        break
