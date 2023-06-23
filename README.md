# UniS-MMC
Code for [UniS-MMC: Multimodal Classification via Unimodality-supervised Multimodal Contrastive Learning](https://arxiv.org/abs/2305.09299) (ACL 2023 Findings)

## Environemt
Python=3.8, Pytorch=1.8.0, CUDA=11.1
```
conda create -n unis111 python=3.8
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## Code Structure
```
-- model: TextEncoder.py, ImageEncoder.py, model.py
-- data: dataloader.py, create_data.py
-- results: logs, results, imgs, models
-- Pretrained: bert_base_uncased, bert_large_uncased, roberta_base, roberta_large, vit_base, vit_large
-- src: config.py, functions.py, metrics.py, train_food101.py
-- main.py
-- train.sh
-- test.sh
-- requirements.txt
```

## Data Preparation
[UPMC-Food-101](https://visiir.isir.upmc.fr/explore) is a multimodal food classification dataset. We adopt the most commonly used split method and remove those image-text pairs with missing images or text. The final dataset split is available [here](https://drive.google.com/drive/folders/11U1pjjQ5z6NaG9Gojo6QrSbIqEMYft7m?usp=share_link).

[N24News](https://github.com/billywzh717/n24news) is a multimodal news classification dataset. We adopt the original split method.

## Train and Test

The examples for training and test are included in train.sh and test.sh. For direct inference on Food101, we provide a pretrained checkpoint [here](https://drive.google.com/file/d/1a46kflmEOSx9sU3mt8lR9CVlTcmwnlWB/view?usp=sharing).


## Citations
If you use this code as part of any published research, we'd really appreciate it if you could cite the following paper:
```text
@inproceedings{Zou2023UniSMMCMC,
  title={UniS-MMC: Multimodal Classification via Unimodality-supervised Multimodal Contrastive Learning},
  author={Heqing Zou and Meng Shen and Chen Chen and Yuchen Hu and Deepu Rajan and Eng Siong Chng},
  year={2023}
}
```

## License

MIT License
