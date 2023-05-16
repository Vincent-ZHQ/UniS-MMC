import os
import argparse

from src.functions import Storage


class Config():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'MMC': self.__MMC
        }

        # normalize
        dataset_name = str.lower(args.dataset)
        # load params
        commonArgs = HYPER_MODEL_MAP['MMC']()['commonParas']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                                 **commonArgs,
                                 **HYPER_MODEL_MAP['MMC']()['datasetParas'][dataset_name],
                                 ))


    def __MMC(self):
        tmp = {
            'commonParas': {
                'need_normalized': False,
                'use_bert': True,
                'use_finetune': True,
            },
            # dataset
            'datasetParas': {
                'food101': {
                    'num_train_data':67972,
                    'early_stop': 4,
                    # training/validation/test parameters
                    'batch_size': 32,
                    'batch_gradient': 128, #60,
                    'max_length': 512,
                    'num_workers': 24,
                    'num_epoch': 25,
                    'patience': 3,
                    'min_epoch': 2,
                    'valid_step': 50, #100,

                    'lr_text_tfm': 5e-5, #2e-5,
                    'lr_img_tfm': 5e-5,#
                    'lr_text_cls': 1e-4, # 5e-5,
                    'lr_img_cls': 1e-4,
                    'lr_mm_cls': 1e-4,
                    'lr_warmup': 0.1,
                    'lr_factor': 0.2,
                    'lr_patience': 2,
                    'weight_decay_tfm': 1e-4,
                    'weight_decay_other': 1e-4,

                    'text_out': 768,
                    'img_out': 768,
                    'post_dim': 256,
                    'output_dim': 101,

                    'text_dropout': 0.1,
                    'img_dropout': 0.1,
                    'mm_dropout': 0.0,

                    'text_encoder': 'bert_base',
                    'image_encoder': 'vit_base',
                },
                'n24news': {
                    'num_train_data': 67972,
                    'early_stop': 4,
                    # training/validation/test parameters
                    'batch_size': 32,
                    'batch_gradient': 128,
                    'max_length': 20,
                    # 'gradient_accumulation_steps': 1,
                    'num_workers': 14,
                    'num_epoch': 12,
                    'patience': 8,
                    'min_epoch': 1,
                    'valid_step': 50,

                    'text_type': 'headline',

                    'lr_text_tfm': 2e-5,
                    'lr_img_tfm': 5e-5,  #
                    'lr_text_cls': 5e-5,
                    'lr_img_cls': 1e-4,
                    'lr_mm_cls': 1e-4,
                    'lr_warmup': 0.1,
                    'lr_factor': 0.2,
                    'lr_patience': 1,
                    'weight_decay_tfm': 1e-4,
                    'weight_decay_other': 1e-4,

                    'text_out': 768,
                    'img_out': 768,
                    'post_dim': 256,
                    'output_dim': 24,

                    'text_dropout': 0.0,  # From 0.1 -> 0
                    'img_dropout': 0.1,
                    'mm_dropout': 0.0,

                    'text_encoder': 'bert_base',
                    'image_encoder': 'vit_base',
                },
                'rosmap': {
                    # training/validation/test parameters
                    'num_epoch': 150,
                    'patience': 25,

                    'lr_mm': 2e-3, # 5e-3
                    'weight_decay_other': 1e-3,

                    'post_dim': 1000,
                    'output_dim': 2,
                    'mm_dropout': 0.5, #0.1,
                },
                'brca': {
                    # training/validation/test parameters
                    'num_epoch': 150,
                    'patience': 25,

                    'lr_mm': 5e-3, # 5e-4
                    'weight_decay_other': 1e-3,

                    'post_dim': 1000,
                    'output_dim': 5,
                    'mm_dropout': 0.2,
                },
            },
        }
        return tmp

    def get_config(self):
        return self.args