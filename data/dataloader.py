import re
import os
import time
import argparse
import random
import logging
import json
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler

from model.TextEncoder import TextEncoder
from model.ImageEncoder import ImageEncoder


__all__ = ['MMDataLoader']

logger = logging.getLogger('MMC')

TAG_RE = re.compile(r'<[^>]+>')


def remove_tags(text):
    return TAG_RE.sub('', text)


def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = sentence.lower()
    return sentence

# Data Aug
def get_transforms():
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
        ]
    )


def remove_tags(text):
    return TAG_RE.sub('', text)


# vec_load_image = np.vectorize(load_image, signature='()->(r,c,d),(s)')
def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = sentence.lower()
    return sentence


def format_txt_file(content):
    for c in '<>/\\+=-_[]{}\'\";:.,()*&^%$#@!~`':
        content = content.replace(c, ' ')
    content = re.sub("\s\s+" , ' ', content)
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = re.sub(r"\s+[a-zA-Z]\s+", ' ', content)
    return content.lower().replace("\n", " ")


label_n24news = { 'Health': 0,
                  'Books': 1,
                  'Science': 2,
                  'Art & Design': 3,
                  'Television': 4,
                  'Style': 5,
                  'Travel': 6,
                  'Media': 7,
                  'Movies': 8,
                  'Food': 9,
                  'Dance': 10,
                  'Well': 11,
                  'Real Estate': 12,
                  'Fashion & Style': 13,
                  'Economy': 14,
                  'Technology': 15,
                  'Sports': 16,
                  'Your Money': 17,
                  'Theater': 18,
                  'Education': 19,
                  'Opinion': 20,
                  'Automobiles': 21,
                  'Music': 22,
                  'Global Business': 23,
                  }

def truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class MMDataset(Dataset):
    def __init__(self, args, labels):
        self.args = args
        self.labels = labels
        self.save = []
        print(os.path.join(args.data_dir, labels))
        if args.dataset in ['Food101']:
            self.df = pd.read_csv(os.path.join(args.data_dir, labels),
                                  dtype={'id': str, 'text': str, 'annotation': str, 'label': int})
        else:
            self.df = json.load(open(os.path.join(args.data_dir, labels), 'r', encoding='utf8'))

        self.text_tokenizer = TextEncoder(pretrained_dir=args.pretrained_dir, text_encoder=args.text_encoder).get_tokenizer()
        self.image_tokenizer = ImageEncoder(pretrained_dir=args.pretrained_dir, image_encoder=args.image_encoder).get_tokenizer()

        self.img_width = 224
        self.img_height = 224
        self.depth = 3
        self.max_length = args.max_length  # Setup according to the text
        self.transforms = get_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.args.dataset in ['Food101']:
            id, text, annotation, label = self.df.loc[index]
            img_path = self.args.data_dir + '/images/' + self.labels[:-4] + '/' + annotation + '/' + id
            text_path = self.args.data_dir + '/texts_txt/' + annotation + '/' + id.replace(".jpg", ".txt")
            text = format_txt_file(open(text_path).read())
        else:
            if self.args.text_type in ['headline']:
                text = self.df[index]['headline']
            elif self.args.text_type in ['caption']:
                text = self.df[index]['caption']
            elif self.args.text_type in ['abstract']:
                text = self.df[index]['abstract']
            else:
                text = self.df[index]['article']
                if self.args.text_encoder not in ['roberta_base']:
                    text = format_txt_file(text)
            img_path = self.args.data_dir + '/imgs/' + self.df[index]['image_id'] + '.jpg'
            label = label_n24news[self.df[index]['section']]

        # text -> text_token
        text_tokens = self.text_tokenizer(text, max_length=self.max_length, add_special_tokens=True, truncation=True,
                                     padding='max_length', return_tensors="pt")
        image = Image.open(os.path.join(img_path)).convert("RGB")
        image = self.transforms(image)
        img_inputs = self.image_tokenizer(images=image, return_tensors="pt").pixel_values

        if 'roberta' in self.args.text_encoder:
            return img_inputs, text_tokens['input_ids'], 0, text_tokens['attention_mask'], label
        else:
            return img_inputs, text_tokens['input_ids'], text_tokens['token_type_ids'], text_tokens[
                'attention_mask'], label


def MMDataLoader(args):
    if args.dataset in ['Food101']:
        train_data_set = MMDataset(args, 'train.csv')
        train_set, valid_set = torch.utils.data.random_split(train_data_set, [len(train_data_set)-5000, 5000])
        test_set = MMDataset(args, 'test.csv')
    else: 
        train_set = MMDataset(args, 'news/nytimes_train.json')
        valid_set = MMDataset(args, 'news/nytimes_dev.json')
        test_set = MMDataset(args, 'news/nytimes_test.json')

    logger.info(f'Train Dataset: {len(train_set)}')
    logger.info(f'Valid Dataset: {len(valid_set)}')
    logger.info(f'Test Dataset: {len(test_set)}')

    if args.local_rank in [-1]:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                  shuffle=False, pin_memory=False, drop_last=True)
    else:
        train_sampler = DistributedSampler(train_set)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                       sampler=train_sampler, pin_memory=False, drop_last=True)

    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.num_workers,
                       shuffle=False, pin_memory=False, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers,
                       shuffle=False, pin_memory=False, drop_last=True)

    return train_loader, valid_loader, test_loader