DEVICE=cuda:7
BRIGHTNESS_CLAMP=1
BRIGHTNESS_REGULARIZER_COEFF=1.0
CONTRAST_CLAMP=1
PSF_LOSS_WEIGHT=1.0
MASKED_LOSS_WEIGHT=1
L1_LOSS_WEIGHT=1
DA_LOSS_WEIGHT=1
PERCEPTUAL_LOSS_WEIGHT=0.0
PHASE_NOISE_STDDEV=0.0 # Eunsue use 0.2
IMAGE_NOISE_STDDEV=0.0 # Eunsue use 0.01
RESULT_PATH=./example/asset/ckpt/fixImgForm_Lens2_BR1_baseline1500_broadband_daloss1_indePSF_meta10000_1024_f9_IMX304_arr22
# RESULT_PATH=./example/asset/ckpt/debugging
PARAM=./example/asset/config/param_IMX304_synMeta_RGB_pitch395nm_2by2_1024_noBP.py
# PARAM=./example/asset/config/param_camPitch1.85_RGB_pitch395nm_864_semiBroad.py
FENCE_DATASET_DIR=None
# DIRT_RAINDROP_DATASET_TRAIN_DIR=./dataset/leftImg8bit_trainvaltest/leftImg8bit/train
# DIRT_RAINDROP_DATASET_VAL_DIR=./dataset/leftImg8bit_trainvaltest/leftImg8bit/test
DIRT_RAINDROP_DATASET_TRAIN_DIR=./dataset/DIV2K/train
DIRT_RAINDROP_DATASET_VAL_DIR=./dataset/LIU4K_v2_validation_arbitrary
OBSTRUCTION=dirt # single_image, dirt, dirt_fence
BATCH_SIZE=1
SAVE_FREQ=800
LOG_FREQ=100
N_EPOCHS=100
PROPAGATOR=SBL_ASM_sparse
RESIZING_METHOD=area
NORMALIZING_METHOD=original
PHASE_INIT=random

# action:"store true" options:
# --use_dataset
# --use_lens
# --use_network
# --use_perc_loss
# --use_da_loss
# --train_RGB
# --train_broadband
# --spatially_varying_PSF
# --random_init
# --constant_wvl_phase

# python train_metasurface.py \
# python train_metasurface_sep_near_far.py \
# python train_metasurface_full_PSF_sep_near_far.py \
# python train_metasurface.py \
python train_lens.py \
--use_dataset \
--constant_wvl_phase \
--train_broadband \
--use_da_loss \
--normalizing_method $NORMALIZING_METHOD \
--phase_init $PHASE_INIT \
--l1_loss_weight $L1_LOSS_WEIGHT \
--da_loss_weight $DA_LOSS_WEIGHT \
--perceptual_loss_weight $PERCEPTUAL_LOSS_WEIGHT \
--masked_loss_weight $MASKED_LOSS_WEIGHT \
--psf_loss_weight $PSF_LOSS_WEIGHT \
--brightness_clamp $BRIGHTNESS_CLAMP \
--brightness_regularizer_coeff $BRIGHTNESS_REGULARIZER_COEFF \
--contrast_clamp $CONTRAST_CLAMP \
--phase_noise_stddev $PHASE_NOISE_STDDEV \
--image_noise_stddev $IMAGE_NOISE_STDDEV \
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