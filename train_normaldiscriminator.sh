EXPERIMENT_NAME=GAN
RESULT_DIR=./result/GAN
CHECKPOINTS_DIR=./result/GAN/checkpoints/

CUDA_VISIBLE_DEVICES=2,3 \
python train.py \
    --result_dir ${RESULT_DIR} \
    --checkpoints_dir ${CHECKPOINTS_DIR} \
    --batch_size 2 \
    --nEpochs 200 \
    --netD GAN \
    --gpu_ids 2,3