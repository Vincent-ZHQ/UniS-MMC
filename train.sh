# Food101 BERT-Base+ViT-Base
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 29917 main.py --name M25d0_Test --dataset Food101 --mmc UniSMMC --batch_size 32 --lr_mm_cls 2e-5 --mm_dropout 0 --lr_text_tfm 5e-05 --lr_img_tfm 5e-05 --lr_img_cls 1e-4 --lr_text_cls 1e-4 --text_dropout 0.1 --img_dropout 0.1  --seeds 1
