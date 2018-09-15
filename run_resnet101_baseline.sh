# check the enviroment info
nvidia-smi
PYTHON="/root/miniconda3/bin/python"

#network config
NETWORK="resnet101"
METHOD="baseline"
DATASET="cityscapes_train"

#training settings
LEARNING_RATE=1e-2
WEIGHT_DECAY=5e-4
START_ITERS=0
MAX_ITERS=40000
BATCHSIZE=8
INPUT_SIZE='769,769'
USE_CLASS_BALANCE=True
USE_OHEM=False
OHEMTHRES=0.7
OHEMKEEP=0
USE_VAL_SET=False
USE_EXTRA_SET=False

# replace the DATA_DIR with your folder path to the dataset.
DATA_DIR='./dataset/cityscapes'
DATA_LIST_PATH='./dataset/list/cityscapes/train.lst'
RESTORE_FROM='./pretrained_model/resnet101-imagenet.pth'


# Set the Output path of checkpoints, training log.
TRAIN_LOG_FILE="./log/log_train/log_${NETWORK}_${METHOD}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}"	
SNAPSHOT_DIR="./checkpoint/snapshots_${NETWORK}_${METHOD}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}/"


########################################################################################################################
#  Training
########################################################################################################################
$PYTHON -u train.py --network $NETWORK --method $METHOD --random-mirror --random-scale --gpu 0,1,2,3 --batch-size $BATCHSIZE \
  --snapshot-dir $SNAPSHOT_DIR  --num-steps $MAX_ITERS --ohem $USE_OHEM --data-list $DATA_LIST_PATH --weight-decay $WEIGHT_DECAY \
  --input-size $INPUT_SIZE --ohem-thres $OHEMTHRES --ohem-keep $OHEMKEEP --use-val $USE_VAL_SET --use-weight $USE_CLASS_BALANCE \
  --snapshot-dir $SNAPSHOT_DIR --restore-from $RESTORE_FROM --start-iters $START_ITERS --learning-rate $LEARNING_RATE  \
  --use-extra $USE_EXTRA_SET --dataset $DATASET --data-dir $DATA_DIR  > $TRAIN_LOG_FILE 2>&1


# testing settings
TEST_USE_FLIP=False
TEST_USE_MS=False
TEST_STORE_RESULT=False
TEST_BATCHSIZE=4
PREDICT_CHOICE='whole'
WHOLE_SCALE='1'
TEST_RESTORE_FROM="${SNAPSHOT_DIR}CS_scenes_${MAX_ITERS}.pth"


########################################################################################################################
#  Testing
########################################################################################################################
# validation set
TESTDATASET="cityscapes_train"
TEST_SET="val"
TEST_DATA_LIST_PATH="./dataset/list/cityscapes/val.lst"
TEST_LOG_FILE="./log/log_test/log_result_${NETWORK}_${METHOD}_${TEST_SET}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}_${PREDICT_CHOICE}"
TEST_OUTPUT_PATH="./visualize/${NETWORK}_${METHOD}_${TEST_SET}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}/"

$PYTHON -u eval.py --network=$NETWORK --method=$METHOD --batch-size=$TEST_BATCHSIZE --data-dir $DATA_DIR --data-list $TEST_DATA_LIST_PATH --dataset $TESTDATASET \
 --restore-from=$TEST_RESTORE_FROM  --store-output=$TEST_STORE_RESULT --output-path=$TEST_OUTPUT_PATH --input-size $INPUT_SIZE \
 --use-flip=$TEST_USE_FLIP  --use-ms=$TEST_USE_MS --gpu 0,1,2,3 --predict-choice $PREDICT_CHOICE --whole-scale ${WHOLE_SCALE} > $TEST_LOG_FILE 2>&1


# training set
TESTDATASET="cityscapes_train"
TEST_SET="train"
TEST_DATA_LIST_PATH="./dataset/list/cityscapes/train.lst"
TEST_LOG_FILE="./log/log_test/log_result_${NETWORK}_${METHOD}_${TEST_SET}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}_${PREDICT_CHOICE}"
TEST_OUTPUT_PATH="./visualize/${NETWORK}_${METHOD}_${TEST_SET}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}/"

$PYTHON -u eval.py --network=$NETWORK --method=$METHOD --batch-size=$TEST_BATCHSIZE --data-dir $DATA_DIR --data-list $TEST_DATA_LIST_PATH --dataset $TESTDATASET \
 --restore-from=$TEST_RESTORE_FROM  --store-output=$TEST_STORE_RESULT --output-path=$TEST_OUTPUT_PATH --input-size $INPUT_SIZE \
 --use-flip=$TEST_USE_FLIP  --use-ms=$TEST_USE_MS --gpu 0,1,2,3 --predict-choice $PREDICT_CHOICE --whole-scale ${WHOLE_SCALE} > $TEST_LOG_FILE 2>&1


## test set
TEST_STORE_RESULT=True
TESTDATASET="cityscapes_test"
TEST_SET="test"
TEST_DATA_LIST_PATH="./dataset/list/cityscapes/test.lst"
TEST_LOG_FILE="./log/log_test/log_result_${NETWORK}_${METHOD}_${TEST_SET}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}_${PREDICT_CHOICE}"
TEST_OUTPUT_PATH="./visualize/${NETWORK}_${METHOD}_${TEST_SET}_${LEARNING_RATE}_${WEIGHT_DECAY}_${BATCHSIZE}_${MAX_ITERS}/"

$PYTHON -u generate_submit.py --network=$NETWORK --method=$METHOD --batch-size=$TEST_BATCHSIZE --data-dir $DATA_DIR --data-list $TEST_DATA_LIST_PATH --dataset $TESTDATASET \
 --restore-from=$TEST_RESTORE_FROM  --store-output=$TEST_STORE_RESULT --output-path=$TEST_OUTPUT_PATH --input-size $INPUT_SIZE \
 --use-flip=$TEST_USE_FLIP  --use-ms=$TEST_USE_MS --gpu 0,1,2,3 --predict-choice $PREDICT_CHOICE --whole-scale ${WHOLE_SCALE} > $TEST_LOG_FILE 2>&1
