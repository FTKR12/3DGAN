EXPERIMENT_NAME=PatchGAN
RESULT_DIR=./result/PatchGAN
CHECKPOINTS_DIR=./result/PatchGAN/checkpoints/

CUDA_VISIBLE_DEVICES=0,1 \
python train.py \
    --result_dir $RESULT_DIR \
    --checkpoints_dir $CHECKPOINTS_DIR \
    --batch_size 2 \
    --nEpochs 200 \
    --netD PatchGAN \
    --gpu_ids 0,1 \
