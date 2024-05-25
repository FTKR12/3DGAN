EXPERIMENT_NAME=PatchGAN
RESULT_DIR=./result/PatchGAN
OUTPUT_DIR=./result/PatchGAN/output
CHECKPOINTS_DIR=./result/PatchGAN/checkpoints

CUDA_VISIBLE_DEVICES=0,1 \
python test.py \
    --netD PatchGAN \
    --generatorWeights ${CHECKPOINTS_DIR}/g.pth \
    --output ${OUTPUT_DIR}