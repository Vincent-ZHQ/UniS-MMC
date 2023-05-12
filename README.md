# UniS-MMC
Code for UniS-MMC: Multimodal Classification via Unimodality-supervised Multimodal Contrastive Learning (ACL 2023 Findings)




## Environemt: Python=3.8, Pytorch=1.8.0, CUDA=11.1
conda create -n unis111 python=3.8
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt


Code Structure

-- UniS-MMC
  -- model: TextEncoder.py, ImageEncoder.py, model.py
  -- data: dataloader.py, create_data.py
  -- results: logs, results, imgs, models
  -- Pretrained: bert_base_uncased, bert_large_uncased, roberta_base, roberta_large, vit_base, vit_large
  -- src: config.py, functions.py, metrics.py, train_food101.py
  -- main.py
  -- requirements.txt

## Data Preparation
\href{https://visiir.isir.upmc.fr/explore}{UPMC-Food-101} is multimodal food clssfication dataset.

\href{https://github.com/billywzh717/n24news}{N24News} is a multimodal news classfication dataset. 


## Pretrained Model for Test


