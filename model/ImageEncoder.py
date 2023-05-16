import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ViTFeatureExtractor, ViTModel

__all__ = ['ImageEncoder']

# vit base model from https://huggingface.co/google/vit-base-patch16-224
# vit large model from https://huggingface.co/google/vit-large-patch16-224

class ImageEncoder(nn.Module):
    def __init__(self, pretrained_dir, image_encoder='base'):
        """
        image_encoder: base / large
        """
        super(ImageEncoder, self).__init__()

        assert image_encoder in ['vit_base', 'vit_large']

        tokenizer = ViTFeatureExtractor
        model = ViTModel
        # directory is fine
        if image_encoder in ['vit_base']:
            self.tokenizer = tokenizer.from_pretrained(pretrained_dir+'/vit_base_224/')
            self.model = model.from_pretrained(pretrained_dir+'/vit_base_224/')
        else:
            self.tokenizer = tokenizer.from_pretrained(pretrained_dir+'/vit_large_224/')
            self.model = model.from_pretrained(pretrained_dir+'/vit_large_224/')

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, pixel_values):
        """
        pixel_values:
        """
        last_hidden_states = self.model(pixel_values=pixel_values).last_hidden_state

        return last_hidden_states


if __name__ == "__main__":
    vit_normal = ImageEncoder()
