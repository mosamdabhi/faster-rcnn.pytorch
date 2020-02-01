CUDA_VISIBLE_DEVICES=0,1,2,3 nice -10 python trainval_net.py \
                   --dataset freihand --net res101 \
                   --bs 1 --nw 10 \
                   --lr 1e-3  --lr_decay_step 5 --epochs 7\
                   --cuda --mGPUs
