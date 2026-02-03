DEVICE=cuda:3
RESULT_PATH=./example/asset/ckpt/area_doe1600_512_debug_doe_network
PARAM=./example/asset/config/param_MV_1600_512.py
PRETRAINED_DOE=./example/asset/ckpt/area_doe1600_512_color_psf/DOE_phase_001.pt
PRETRAINED_NETWORK=./example/asset/ckpt/area_doe1600_512_network/network_001.pt
FENCE_DATASET_DIR=None
DIRT_RAINDROP_DATASET_TRAIN_DIR=./dataset/leftImg8bit_trainvaltest/leftImg8bit/train
DIRT_RAINDROP_DATASET_VAL_DIR=./dataset/leftImg8bit_trainvaltest/leftImg8bit/test
OBSTRUCTION=dirt_raindrop
BATCH_SIZE=1
SAVE_FREQ=2900
LOG_FREQ=20
N_EPOCHS=500
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

# --pretrained_DOE $PRETRAINED_DOE \
# --pretrained_network $PRETRAINED_NETWORK \

python train_metasurface.py \
--batch_size $BATCH_SIZE \
--log_freq $LOG_FREQ \
--save_freq $SAVE_FREQ \
--n_epochs $N_EPOCHS \
--resizing_method $RESIZING_METHOD \
--use_dataset \
--use_network \
--color_PSF \
--result_path $RESULT_PATH \
--param_file $PARAM \
--device $DEVICE \
--dirt_raindrop_dataset_train_dir $DIRT_RAINDROP_DATASET_TRAIN_DIR \
--dirt_raindrop_dataset_val_dir $DIRT_RAINDROP_DATASET_VAL_DIR \
--obstruction $OBSTRUCTION \
--propagator $PROPAGATOR