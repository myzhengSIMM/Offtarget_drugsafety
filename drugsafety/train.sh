#! /bin/bash

#  The kinases data set is taken as an example to demonstrate the training process of our model

#for DATASET_NAME in "parsedGPCR_3decoys" "parsedIonchannel_5decoys" "parsedNR_3decoys" "parsedother_5decoys" "parsedEnzyme_3decoys" "parsedKinases_3decoys" "parsedTransporter_5decoys"
for DATASET_NAME in "parsedKinases_3decoys"
do
    if [[ $DATASET_NAME == "parsedEnzyme_3decoys" ]]; then
        NUM_LAYERS=1
        NUM_TIMESTEPS=4
        GRAPH_FEAT_SIZE=600
        BATCH_SIZE=64
        LR=0.0001
        WEIGHT_DECAY=0.0
    elif [[ $DATASET_NAME == "parsedKinases_3decoys" ]]; then
        NUM_LAYERS=3
        NUM_TIMESTEPS=2
        GRAPH_FEAT_SIZE=200
        BATCH_SIZE=64
        LR=0.0005
        WEIGHT_DECAY=1e-7
    elif [[ $DATASET_NAME == "parsedTransporter_5decoys" ]]; then
        PROJECT_DIRECTORY="/home/liujin/Offtarget_drugsafety"
        NUM_LAYERS=3
        NUM_TIMESTEPS=2
        GRAPH_FEAT_SIZE=200
        BATCH_SIZE=64
        LR=0.0005
        WEIGHT_DECAY=0.0
    else
        NUM_LAYERS=3
        NUM_TIMESTEPS=2
        GRAPH_FEAT_SIZE=200
        BATCH_SIZE=128
        LR=0.0001
        WEIGHT_DECAY=0.0
    fi
    
    PROJECT_DIRECTORY="/home/liujin/Offtarget_drugsafety"
    TASK_INSTANCE_UUID=${DATASET_NAME}_$(date +%s)
    SPLIT='random_stratified'

    python ./train.py   --dataset_dir ${PROJECT_DIRECTORY}/databases/model_dataset_new \
                              --dataset_name ${DATASET_NAME} \
                              --result_dir ${PROJECT_DIRECTORY}/drugsafety/results/data_combine_train_new \
                              --project ${TASK_INSTANCE_UUID} \
                              --gpu 0 --split ${SPLIT} \
                              --dropout 0.1  --num_layers ${NUM_LAYERS}  --num_timesteps ${NUM_TIMESTEPS}  --graph_feat_size ${GRAPH_FEAT_SIZE} \
                              --lr ${LR}  --weight_decay ${WEIGHT_DECAY}  --batch_size ${BATCH_SIZE}  --epochs 100 \

done