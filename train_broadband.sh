DEVICE=cuda:6
L1_LOSS_WEIGHT=1
PERC_LOSS_WEIGHT=1
SSIM_LOSS_WEIGHT=1.0
DA_LOSS_WEIGHT=1.0
BRIGHTNESS_REGULARIZER_WEIGHT=3.0
OPTICS_LAYER_LR=0.1
OPTICS_CLASS_LR=0.1
T_MAX=5000
TAU_START=0.5
TAU_END=0.5
HARD_START=0.0
WARM_FRAC=0.0
# RESULT_PATH=../Friday_log/doublet_broadband
RESULT_PATH=./logs/speedtest
PARAM=./params/param_doublet_broadband.py
BATCH_SIZE=1
SAVE_FREQ=500
LOG_FREQ=100
N_EPOCHS=100000
PROPAGATOR=SBL_ASM
RESIZING_METHOD=area
NORMALIZING_METHOD=original
PHASE_INIT=random
IMAGE_DATASET_TRAIN_DIR=./img_dataset/Target
IMAGE_DATASET_VAL_DIR=./img_val_dataset
SQSQ=./library/broadband/sqsq.npy
SQCY=./library/broadband/sqcy.npy
CYCY=./library/broadband/cycy.npy
CYSQ=./library/broadband/cysq.npy

# action:"store true" options:
# --use_transmittance_penalty
# --use_perc_loss
# --use_ssim_loss
# --use_da_loss
# --random_init
# --propagator_linear_convolution

python train_broadband.py \
--use_ssim_loss \
--use_da_loss \
--normalizing_method $NORMALIZING_METHOD \
--phase_init $PHASE_INIT \
--l1_loss_weight $L1_LOSS_WEIGHT \
--perc_loss_weight $PERC_LOSS_WEIGHT \
--ssim_loss_weight $SSIM_LOSS_WEIGHT \
--da_loss_weight $DA_LOSS_WEIGHT \
--brightness_regularizer_weight $BRIGHTNESS_REGULARIZER_WEIGHT \
--optics_layer_lr $OPTICS_LAYER_LR \
--optics_class_lr $OPTICS_CLASS_LR \
--T_max $T_MAX \
--tau_start $TAU_START \
--tau_end $TAU_END \
--hard_start $HARD_START \
--warm_frac $WARM_FRAC \
--log_freq $LOG_FREQ \
--save_freq $SAVE_FREQ \
--n_epochs $N_EPOCHS \
--resizing_method $RESIZING_METHOD \
--result_path $RESULT_PATH \
--param_file $PARAM \
--device $DEVICE \
--propagator $PROPAGATOR \
--training_dir $IMAGE_DATASET_TRAIN_DIR \
--validation_dir $IMAGE_DATASET_VAL_DIR \
--sqsq $SQSQ \
--sqcy $SQCY \
--cycy $CYCY \
--cysq $CYSQ \