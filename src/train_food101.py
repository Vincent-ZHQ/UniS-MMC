import os
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from sklearn.metrics import accuracy_score

from model.model import MMC
from data.dataloader import MMDataLoader
from src.metrics import collect_metrics
from src.functions import save_checkpoint, load_checkpoint, dict_to_str, count_parameters

__all__ = ['TrainModule']

logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger('MMC')


# To decide the lr scheduler
def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


# To decide the optimizer
def get_optimizer(model, args):
    # if args.local_rank in [-1]:
    if args.mmc not in ['V']:
        text_enc_param = list(model.module.text_encoder.named_parameters())
        text_clf_param = list(model.module.text_classfier.parameters())
    if args.mmc not in ['T']:
        img_enc_param = list(model.module.image_encoder.parameters())
        img_clf_param = list(model.module.image_classfier.parameters())
    mm_clf_param = list(model.module.mm_classfier.parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    if args.mmc in ['V']:
        optimizer_grouped_parameters = [
            {"params": img_enc_param, "weight_decay": args.weight_decay_tfm, 'lr': args.lr_img_tfm},
            {"params": img_clf_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_img_cls},
            {"params": mm_clf_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_mm_cls},
        ]
    elif args.mmc in ['T']:
        optimizer_grouped_parameters = [
            {"params": [p for n, p in text_enc_param if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay_tfm, 'lr': args.lr_text_tfm},
            {"params": [p for n, p in text_enc_param if any(nd in n for nd in no_decay)], "weight_decay": 0.0,
             'lr': args.lr_text_tfm},
            {"params": text_clf_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_text_cls},
            {"params": mm_clf_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_mm_cls},
        ]
    else:
        optimizer_grouped_parameters = [
            {"params": [p for n, p in text_enc_param if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay_tfm, 'lr': args.lr_text_tfm},
            {"params": [p for n, p in text_enc_param if any(nd in n for nd in no_decay)], "weight_decay": 0.0,
             'lr': args.lr_text_tfm},
            {"params": text_clf_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_text_cls},
            {"params": img_enc_param, "weight_decay": args.weight_decay_tfm, 'lr': args.lr_img_tfm},
            {"params": img_clf_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_img_cls},
            {"params": mm_clf_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_mm_cls},
        ]
    optimizer = optim.Adam(optimizer_grouped_parameters)

    return optimizer


def valid(args, model, data=None, best_valid=None, nBetter=None, step=None):
    model.eval()
    with torch.no_grad():
        train_loader, valid_loader, test_loader = data
        y_pred = []
        y_true = []
        with tqdm(valid_loader) as td:
            for batch_image, text_input_ids, text_token_type_ids, text_attention_mask, batch_label in td:
                text = text_input_ids.to(args.device), text_token_type_ids.to(args.device), text_attention_mask.to(args.device)
                image = batch_image.to(args.device)
                logit = model.module.infer(text, image, None)
                y_pred.append(logit.cpu())
                y_true.append(batch_label.cpu())
                # break
        logits = torch.cat(y_pred)
        te_true = torch.cat(y_true).data.cpu().numpy()
        te_prob = F.softmax(logits, dim=1).data.cpu().numpy()
        cur_valid = accuracy_score(te_true, te_prob.argmax(1))
        isBetter = cur_valid >= (best_valid + 1e-6)
        valid_results = {"step": step}
        valid_results.update(collect_metrics(args.dataset, te_true, te_prob))
        if isBetter:
            if args.local_rank in [0, -1]:
                save_checkpoint(model, args.best_model_save_path)
            best_valid = cur_valid
            nBetter = 0
        else:
            nBetter += 1
        return valid_results, best_valid, nBetter


def train_valid(args, model, optimizer, scheduler=None, data=None):
    model.train()
    best_valid = 1e-5
    nBetter = 0
    train_loss_m = 0
    total_step = 0
    gradient_accumulation_steps = int(args.batch_gradient / args.batch_size)
    for epoch in range(args.num_epoch + 1):
        train_loader, valid_loader, test_loader = data
        y_pred = []
        y_true = []
        if args.local_rank not in [-1]:
            train_loader.sampler.set_epoch(epoch)
        with tqdm(train_loader) as td:
            for batch_image, text_input_ids, text_token_type_ids, text_attention_mask, batch_label in td:
                text = text_input_ids.to(args.device), text_token_type_ids.to(args.device), text_attention_mask.to(args.device)
                image = batch_image.to(args.device)
                labels = batch_label.to(args.device).view(-1)
                # optimizer.zero_grad()
                loss, loss_m, logit_m = model(text, image, None, labels)
                # print(loss)
                loss = loss.sum() # / gradient_accumulation_steps
                loss.backward()
                # optimizer.step()
                train_loss_m += loss_m.sum().item()
                y_pred.append(logit_m.cpu())
                y_true.append(batch_label.cpu())
                total_step += 1

                if total_step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                if (total_step % (args.valid_step * gradient_accumulation_steps) == 0) and (epoch > args.min_epoch):
                # if total_step % args.valid_step == 0:
                    valid_results, best_valid, nBetter = valid(args, model, data, best_valid, nBetter, total_step)
                    if nBetter < 1:
                        if args.local_rank in [-1, 0]:
                            logger.info(args.dataset + " Valid: " + dict_to_str(valid_results))
                        best_results = valid_results
                    if nBetter > args.patience:
                        return best_results
                    # print(args.dataset + " Valid: " + dict_to_str(valid_results))
                    # return best_results
            logits = torch.cat(y_pred)
            tr_true = torch.cat(y_true).data.cpu().numpy()
            tr_prob = F.softmax(logits, dim=1).data.cpu().numpy()
            tuning_metric = accuracy_score(tr_true, tr_prob.argmax(1))
            scheduler.step(tuning_metric)
    return best_results


def test_epoch(model, dataloader=None):
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        with tqdm(dataloader) as td:
            for batch_image, text_input_ids, text_token_type_ids, text_attention_mask, batch_label in td:
                text = text_input_ids.cuda(), text_token_type_ids.cuda(), text_attention_mask.cuda()
                image = batch_image.cuda()
                logit = model.module.infer(text, image, None)
                y_pred.append(logit.cpu())
                y_true.append(batch_label.cpu())
        logits = torch.cat(y_pred)
        true = torch.cat(y_true).data.cpu().numpy()
        prob = F.softmax(logits, dim=1).data.cpu().numpy()
    return prob, true


def train_food101(args):
    train_loader, valid_loader, test_loader = MMDataLoader(args)
    data = train_loader, valid_loader, test_loader

    if args.local_rank in [-1]:
        model = DataParallel(MMC(args))
        model = model.to(args.device)
    else:
        model = MMC(args).to(args.device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                    output_device=args.local_rank,
                                                    find_unused_parameters=True)
    if args.local_rank in [-1, 0]:
        logger.info(f'\nThe model has {count_parameters(model)} trainable parameters')
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    if not args.test_only:
        if args.local_rank in [-1, 0]:
            logger.info("Start training...")
        best_results = train_valid(args, model, optimizer, scheduler, data)

    load_checkpoint(model, args.best_model_save_path)
    te_prob, te_true = test_epoch(model, test_loader)
    best_results = collect_metrics(args.dataset, te_true, te_prob)
    if args.local_rank in [-1, 0]:
        logger.info("Test: " + dict_to_str(collect_metrics(args.dataset, te_true, te_prob)))

    return best_results #, value0, value1, value2
