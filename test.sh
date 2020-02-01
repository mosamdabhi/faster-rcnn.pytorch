CUDA_VISIBLE_DEVICES=2 python test_net.py --dataset freihand --net res101 \
                   --checksession 1 --checkepoch 7 --checkpoint 9999 \
                   --cuda
