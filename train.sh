CUDA_VISIBLE_DEVICES=2 python trainval_net.py \
                   --dataset freihand --net res101 \
                   --bs 1 --nw 10 \
                   --lr 1e-3  --lr_decay_step 5 --epochs 1\
                   --cuda --mGPUs
