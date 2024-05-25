EXPERIMENT_NAME=PixelGAN
RESULT_DIR=./result/PixelGAN
CHECKPOINTS_DIR=./result/PixelGAN/checkpoints/

CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --result_dir ${RESULT_DIR} \
    --checkpoints_dir ${CHECKPOINTS_DIR} \
    --batch_size 1 \
    --nEpochs 200 \
    --netD PixelGAN \
    --gpu_ids 0