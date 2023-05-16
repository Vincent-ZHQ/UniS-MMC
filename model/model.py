import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.TextEncoder import TextEncoder
from model.ImageEncoder import ImageEncoder

__all__ = ['MMC']


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

class MMC(nn.Module):
    def __init__(self, args):
        super(MMC, self).__init__()
        # text subnets
        self.args = args
        if self.args.mmc not in ['T']:
            self.image_encoder = ImageEncoder(pretrained_dir=args.pretrained_dir, image_encoder=args.image_encoder)
            self.image_classfier = Classifier(args.img_dropout, args.img_out, args.post_dim, args.output_dim)
        if self.args.mmc not in ['V']:
            self.text_encoder = TextEncoder(pretrained_dir=args.pretrained_dir, text_encoder=args.text_encoder)
            self.text_classfier = Classifier(args.text_dropout, args.text_out, args.post_dim, args.output_dim)
        self.mm_classfier = Classifier(args.mm_dropout, args.text_out + args.img_out, args.post_dim, args.output_dim)

    def forward(self, text=None, image=None, data_list=None, label=None, infer=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        feature_um = dict()
        output_um = dict()
        UMLoss = dict()

        text = self.text_encoder(text=text)
        image = torch.squeeze(image, 1)
        image = self.image_encoder(pixel_values=image)
        output_text = self.text_classfier(text[:, 0, :])
        output_image = self.image_classfier(image[:, 0, :])

        fusion = torch.cat([text[:, 0, :], image[:, 0, :]], dim=-1)
        output_mm = self.mm_classfier(fusion)

        if infer:
            return output_mm

        MMLoss_m = torch.mean(criterion(output_mm, label))

        if self.args.mmc in ['NoMMC']:
            MMLoss_sum = MMLoss_m
            return MMLoss_sum, MMLoss_m, output_mm

        if self.args.mmc in ['SupMMC']:
            mmcLoss = self.mmc_2(text[:, 0, :], image[:, 0, :], None, None, label)
            MMLoss_sum = MMLoss_m + 0.1 * mmcLoss
            return MMLoss_sum, MMLoss_m, output_mm

        if self.args.mmc in ['UnSupMMC']:
            mmcLoss = self.mmc_2(text[:, 0, :], image[:, 0, :], None, None, None)
            MMLoss_sum = MMLoss_m + 0.1 * mmcLoss
            return MMLoss_sum, MMLoss_m, output_mm

        MMLoss_text = torch.mean(criterion(output_text, label))
        MMLoss_image = torch.mean(criterion(output_image, label))
        mmcLoss = self.mmc_2(text[:, 0, :], image[:, 0, :], output_text, output_image, label)
        MMLoss_sum = MMLoss_text + MMLoss_image + MMLoss_m + 0.1 * mmcLoss

        return MMLoss_sum, MMLoss_m, output_mm

    def infer(self, text=None, image=None, data_list=None):
        MMlogit = self.forward(text, image, data_list, infer=True)
        return MMlogit

    def mmc_2(self, f0, f1, p0, p1, l):
        f0 = f0 / f0.norm(dim=-1, keepdim=True)
        f1 = f1 / f1.norm(dim=-1, keepdim=True)

        if p0 is not None:
            p0 = torch.argmax(F.softmax(p0, dim=1), dim=1)
            p1 = torch.argmax(F.softmax(p1, dim=1), dim=1)

        if l is None:
            return self.UnSupMMConLoss(f0, f1)
        elif p0 is None:
            return self.SupMMConLoss(f0, f1, l)
        else:
            return self.UniSMMConLoss(f0, f1, p0, p1, l)

    def UniSMMConLoss(self, feature_a, feature_b, predict_a, predict_b, labels, temperature=0.07):
        feature_a_ = feature_a.detach()
        feature_b_ = feature_b.detach()

        a_pre = predict_a.eq(labels)  # a True or not
        a_pre_ = ~a_pre
        b_pre = predict_b.eq(labels)  # b True or not
        b_pre_ = ~b_pre

        a_b_pre = torch.gt(a_pre | b_pre, 0)  # For mask ((P: TT, nP: TF & FT)=T, (N: FF)=F)
        a_b_pre_ = torch.gt(a_pre & b_pre, 0) # For computing nP, ((P: TT)=T, (nP: TF & FT, N: FF)=F)

        a_ = a_pre_ | a_b_pre_  # For locating nP not gradient of a
        b_ = b_pre_ | a_b_pre_  # For locating nP not gradient of b

        if True not in a_b_pre:
            a_b_pre = ~a_b_pre
            a_ = ~a_
            b_ = ~b_
        mask = a_b_pre.float()
#
        feature_a_f = [feature_a[i].clone() for i in range(feature_a.shape[0])]
        for i in range(feature_a.shape[0]):
            if not a_[i]:
                feature_a_f[i] = feature_a_[i].clone()
        feature_a_f = torch.stack(feature_a_f)

        feature_b_f = [feature_b[i].clone() for i in range(feature_b.shape[0])] # feature_b  # [[0,1]])
        for i in range(feature_b.shape[0]):
            if not b_[i]:
                feature_b_f[i] = feature_b_[i].clone()
        feature_b_f = torch.stack(feature_b_f)

        # compute logits
        logits = torch.div(torch.matmul(feature_a_f, feature_b_f.T), temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)

        # compute log_prob
        exp_logits = torch.exp(logits-logits_max.detach())[0]
        mean_log_pos = - torch.log(((mask * exp_logits).sum() / exp_logits.sum()) / mask.sum())# + 1e-6

        return mean_log_pos

    def SupMMConLoss(self, feature_a, feature_b, labels, temperature=0.07):
        # compute the mask matrix
        labels = labels.contiguous().view(-1, 1)
        # mask = torch.eq(labels, labels.T).float() - torch.eye(feature_a.shape[0], feature_a.shape[0])
        mask = torch.eq(labels, labels.T).float()

        # compute logits
        logits = torch.div(torch.matmul(feature_a, feature_b.T), temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_pos = -(mask * log_prob).sum(1) / mask.sum(1)

        return mean_log_pos.mean()

    def UnSupMMConLoss(self, feature_a, feature_b, temperature=0.07):

        # compute the mask matrix
        mask = torch.eye(feature_a.shape[0], dtype=torch.float32).to(self.args.device)

        # compute logits
        logits = torch.div(torch.matmul(feature_a, feature_b.T), temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_pos = -(mask * log_prob).sum(1) / mask.sum(1)
        mean_log_pos = mean_log_pos.mean()

        return mean_log_pos


class Classifier(nn.Module):
    def __init__(self, dropout, in_dim, post_dim, out_dim):
        super(Classifier, self).__init__()
        self.post_dropout = nn.Dropout(p=dropout)
        self.post_layer_1 = LinearLayer(in_dim, post_dim)
        self.post_layer_2 = LinearLayer(post_dim, post_dim)
        self.post_layer_3 = LinearLayer(post_dim, out_dim)

    def forward(self, input):
        input_p1 = F.relu(self.post_layer_1(input), inplace=False)
        input_d = self.post_dropout(input_p1)
        input_p2 = F.relu(self.post_layer_2(input_d), inplace=False)
        output = self.post_layer_3(input_p2)
        return output






