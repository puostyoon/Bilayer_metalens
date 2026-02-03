PRETRAINED_G=None
PRETRAINED_DOE=None
RESULT_DIR=ckpt/E2E_MV2400
PARAM=example/asset/ckpts/MV_recon/param_MV_2400.py
OBSTRUCTION=dirt_raindrop
RANDOM_INIT=--random_init


python train_1D.py --train_optics --result_path $RESULT_DIR --param_file $PARAM --obstruction $OBSTRUCTION --pretrained_DOE $PRETRAINED_DOE --pretrained_G $PRETRAINED_G $RANDOM_INIT