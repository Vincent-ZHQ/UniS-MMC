# Food101 BERT-Base+ViT-Base
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 29926 main.py --name M25d0 --dataset Food101 --mmc UniSMMC --batch_size 32 --seeds 1 --test_only True
