DEVICE=cuda:0
RESULT_PATH=/root/data1/ckpt/deeplabv3plus_daloss_5e-1_meta12000
PARAM=./example/asset/config/param_MV_12000_metasurface.py
FENCE_DATASET_DIR=None
DIRT_RAINDROP_DATASET_TRAIN_DIR=./dataset/leftImg8bit_trainvaltest/leftImg8bit/train
DIRT_RAINDROP_DATASET_VAL_DIR=./dataset/leftImg8bit_trainvaltest/leftImg8bit/test
OBSTRUCTION=dirt
BATCH_SIZE=1
SAVE_FREQ=2900
LOG_FREQ=20
N_EPOCHS=1
PROPAGATOR=Fraunhofer
RESIZING_METHOD=area

# action:"store true" options:
# --use_dataset
# --use_lens
# --use_network
# --use_perc_loss
# --use_da_loss
# --color_PSF
# --random_init

python train_metasurface_da_loss_segmentation.py \
--batch_size $BATCH_SIZE \
--log_freq $LOG_FREQ \
--save_freq $SAVE_FREQ \
--n_epochs $N_EPOCHS \
--resizing_method $RESIZING_METHOD \
--use_dataset \
--result_path $RESULT_PATH \
--param_file $PARAM \
--device $DEVICE \
--dirt_raindrop_dataset_train_dir $DIRT_RAINDROP_DATASET_TRAIN_DIR \
--dirt_raindrop_dataset_val_dir $DIRT_RAINDROP_DATASET_VAL_DIR \
--obstruction $OBSTRUCTION \
--propagator $PROPAGATOR
