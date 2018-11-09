CUDA_VISIBLE_DEVICES=0 python PCB.py -d market -a resnet50 -b 64 -j 4 --log logs/market-1501/PCBRPP_g_20epoch/  --feature 2048 --width 128 --epochs 20 --step-size 20 --data-dir ~/share2/data/Market-1501-v15.09.15/ --height 384

CUDA_VISIBLE_DEVICES=0 python RPP.py -d market -a PCBRPP_g -b 64 -j 4 --log logs/market-1501/PCBRPP_g/  --feature 2048 --height 384 --width 128 --epochs 50 --step_size 20 --data-dir ~/share2/data/Market-1501-v15.09.15/ --resume logs/market-1501/PCBRPP_g_20epoch/checkpoint.pth.tar
