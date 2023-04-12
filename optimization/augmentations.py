import torch
from torch import nn
import kornia.augmentation as K
# import ipdb

class ImageAugmentations(nn.Module):
    def __init__(self, output_size, aug_prob, p_min, p_max, patch=False):
        super().__init__()
        self.output_size = output_size
        
        self.aug_prob = aug_prob
        self.patch = patch
        self.augmentations = nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=aug_prob, padding_mode="border"),  # type: ignore
            K.RandomPerspective(0.7, p=aug_prob),
        )
        self.random_patch = K.RandomResizedCrop(size=(128,128), scale=(p_min,p_max))
        self.avg_pool = nn.AdaptiveAvgPool2d((self.output_size, self.output_size))

    def forward(self, input, num_patch=None, is_global=False):
        """Extents the input batch with augmentations

        If the input is consists of images [I1, I2] the extended augmented output
        will be [I1_resized, I2_resized, I1_aug1, I2_aug1, I1_aug2, I2_aug2 ...]

        Args:
            input ([type]): input batch of shape [batch, C, H, W]

        Returns:
            updated batch: of shape [batch * augmentations_number, C, H, W]
        """
        if self.patch:
            if is_global:
                input = input.repeat(num_patch,1,1,1)
            else:
                input_patches = []
                for i in range(num_patch):
                    if self.aug_prob > 0.0:
                        tmp = self.augmentations(self.random_patch(input))
                    else:
                        tmp = self.random_patch(input)
                    input_patches.append(tmp)
                input = torch.cat(input_patches,dim=0)
        
        else:
            input_patches = []
            for i in range(num_patch):
                input_patches.append(self.augmentations(input))
            input = torch.cat(input_patches,dim=0)
        
        resized_images = self.avg_pool(input)
        return resized_images

    

