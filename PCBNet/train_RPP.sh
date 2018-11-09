CUDA_VISIBLE_DEVICES=0 python PCB.py -d market -a resnet50 -b 64 -j 4 --log logs/market-1501/PCB_20epoch_g/ --height 384 --width 128 --epochs 20 --step_size 20 --data-dir ~/share2/data/Market-1501-v15.09.15/ --feature 2048


CUDA_VISIBLE_DEVICES=0 python RPP.py -d market -a resnet50_rpp -b 64 -j 4 --log logs/market-1501/RPP_g/  --height 384 --width 128 --epochs 50 --step_size 20 --data-dir ~/share2/data/Market-1501-v15.09.15/ --resume logs/market-1501/PCB_20epoch_g/checkpoint.pth.tar --feature 2048
