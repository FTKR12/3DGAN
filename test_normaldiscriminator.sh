EXPERIMENT_NAME=GAN
RESULT_DIR=./result/GAN
OUTPUT_DIR=./result/GAN/output
CHECKPOINTS_DIR=./result/GAN/checkpoints

CUDA_VISIBLE_DEVICES=2,3 \
python test.py \
    --netD GAN \
    --generatorWeights ${CHECKPOINTS_DIR}/g.pth \
    --output ${OUTPUT_DIR}