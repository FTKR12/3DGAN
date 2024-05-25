EXPERIMENT_NAME=test
RESULT_DIR=./result/$EXPERIMENT
CHECKPOINTS_DIR=./result/$EXPERIMENT/checkpoints/

python train.py \
    --result_dir $RESULT_DIR \
    --checkpoints_dir $CHECKPOINTS_DIR \
    --batch_size 2 \
    --nEpochs 2 \