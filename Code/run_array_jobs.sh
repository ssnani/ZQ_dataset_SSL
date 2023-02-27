#!/bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3'
srun python DC_CRN_train.py --num_nodes=$1 \
                  --num_gpu_per_node=$2 \
                  --T60=0.0 \
                  --SNR=-5.0 \
                  --dataset_dtype=stationary \
                  --dataset_condition=noisy_reverb \
                  --noise_simulation=diffuse \
                  --diffuse_files_path=None \
                  --ref_mic_idx=-2 \
                  --train \
                  --dataset_file=/fs/scratch/PAS0774/Shanmukh/Databases/ZQ_LOC_Dataset/features_32_8_ms_20cm/trainset \
                  --val_dataset_file=/fs/scratch/PAS0774/Shanmukh/Databases/ZQ_LOC_Dataset/features_32_8_ms_20cm/devset \
                  --bidirectional \
                  --batch_size=32 \
                  --max_n_epochs=100 \
                  --num_workers=4 \
                  --ckpt_dir=/fs/scratch/PAS0774/Shanmukh/Experiments/ZQ_Dataset_Results/DC_CRN/ \
                  --exp_name=With_Input_Norm  \
                  --resume_model=last.ckpt \
                  --net_type=SISO \
                  --array_job \
                  --input_train_filename=$3