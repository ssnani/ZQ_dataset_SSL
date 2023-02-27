#!/bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3'
srun python ZQ_train.py --num_nodes=$1 \
                  --num_gpu_per_node=$2 \
                  --T60=0.0 \
                  --SNR=-5.0 \
                  --dataset_dtype=stationary \
                  --dataset_condition=noisy_reverb \
                  --noise_simulation=diffuse \
                  --diffuse_files_path=/scratch/bbje/battula12/Databases/Timit/train_spk_signals \
                  --ref_mic_idx=-2 \
                  --dataset_file=/fs/scratch/PAS0774/Shanmukh/Databases/ZQ_LOC_Dataset/features_32_8_ms_20cm/testset_habet-Kun \
                  --val_dataset_file=../test_dataset_file_real_rir_circular_motion.txt \
                  --bidirectional \
                  --batch_size=1 \
                  --num_workers=1 \
                  --ckpt_dir=/fs/scratch/PAS0774/Shanmukh/Experiments/ZQ_Results/Linear_2mic_20cm/PSM_MSE/stationary \
                  --exp_name=With_Input_Norm \
                  --net_type=SISO \
                  --model_path=epoch=21-step=4290.ckpt \
                  --input_test_filename=$3