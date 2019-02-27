CUDA_VISIBLE_DIVICES="1,2,3"
ARCH="ADFLNet"
DATASET="vehicleid_v1.0"
BATCH_SIZE=32
EPOCHS=40
TEST_SIZE=800
LOSS="Triplet"
LOGS_DIR="logs/ADFLNet_Triplet"
python train_adfnet.py -a $ARCH -l $LOSS -d $DATASET -b $BATCH_SIZE --epochs $EPOCHS --test_size $TEST_SIZE --logs $LOGS_DIR
