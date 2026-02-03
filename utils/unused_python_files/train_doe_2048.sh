DEVICE=cuda:7
BRIGHTNESS_CLAMP=1.2
PHASE_NOISE_STDDEV=0.0 # Eunsue use 0.2
IMAGE_NOISE_STDDEV=0.0 # Eunsue use 0.01
RESULT_PATH=./example/asset/ckpt/doe2980_2048_changeWvlDebug_new_normalize_f20mm_clamp1.2
PARAM=./example/asset/config/param_MV_2980_2048.py
FENCE_DATASET_DIR=None
DIRT_RAINDROP_DATASET_TRAIN_DIR=./dataset/leftImg8bit_trainvaltest/leftImg8bit/train
DIRT_RAINDROP_DATASET_VAL_DIR=./dataset/leftImg8bit_trainvaltest/leftImg8bit/test
OBSTRUCTION=dirt
BATCH_SIZE=1
SAVE_FREQ=100
LOG_FREQ=20
N_EPOCHS=50
PROPAGATOR=SBL_ASM_sparse
RESIZING_METHOD=area

# action:"store true" options:
# --use_dataset
# --use_lens
# --use_network
# --use_perc_loss
# --use_da_loss
# --train_RGB
# --spatially_varying_PSF
# --random_init
# --constant_wvl_phase

python train_metasurface.py \
--use_dataset \
--constant_wvl_phase \
--phase_noise_stddev $PHASE_NOISE_STDDEV \
--image_noise_stddev $IMAGE_NOISE_STDDEV \
--brightness_clamp $BRIGHTNESS_CLAMP \
--batch_size $BATCH_SIZE \
--log_freq $LOG_FREQ \
--save_freq $SAVE_FREQ \
--n_epochs $N_EPOCHS \
--resizing_method $RESIZING_METHOD \
--result_path $RESULT_PATH \
--param_file $PARAM \
--device $DEVICE \
--dirt_raindrop_dataset_train_dir $DIRT_RAINDROP_DATASET_TRAIN_DIR \
--dirt_raindrop_dataset_val_dir $DIRT_RAINDROP_DATASET_VAL_DIR \
--obstruction $OBSTRUCTION \
--propagator $PROPAGATOR