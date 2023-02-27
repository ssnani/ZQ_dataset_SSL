import argparse

# parse the configurations
ZQ_parser = argparse.ArgumentParser(description='Additioal configurations for training',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

ZQ_parser.add_argument('--ckpt_dir',
                    type=str,
                    required=True,
                    help='Name of the directory to dump checkpoint')
ZQ_parser.add_argument('--exp_name',
                    type=str,
                    required=True,
                    help='Unit of sample, can be either `seg` or `utt`')

# Experiment parameters (Dataset)
ZQ_parser.add_argument('--T60',
                    type=float,
                    default=0.0,
                    help='T60') 

ZQ_parser.add_argument('--SNR',
                    type=float,
                    default=0.0,
                    help='SNR') 

ZQ_parser.add_argument('--dataset_dtype',
                    type=str,
                    default='',
                    help='moving vs stationary')

ZQ_parser.add_argument('--dataset_condition',
                    type=str,
                    default='',
                    help='["ideal", "noisy", "reverb", "noisy_reverb"]')     

ZQ_parser.add_argument('--noise_simulation',
                    type=str,
                    default='',
                    help='point_source vs diffuse')

ZQ_parser.add_argument('--diffuse_files_path',
                    type=str,
                    default='',
                    help='path of file containing noi files for diffuse simulation')          

ZQ_parser.add_argument('--ref_mic_idx',
                    type=int,
                    default=0,
                    help='ref mic idx 0 or 1 or -1(MIMO) or -2 (SISO)') 

ZQ_parser.add_argument('--train',
                    action='store_true',
                    help='Trained model to test')


# Dataset Files
ZQ_parser.add_argument('--dataset_file',
                    type=str,
                    required=True,
                    help='Train dataset files path (*.pkl) '
                    )

ZQ_parser.add_argument('--val_dataset_file',
                    type=str,
                    required=True,
                    help='Validation dataset files path (*.pkl) '
                    )

# Network hyper parameters
ZQ_parser.add_argument('--bidirectional',
                    action='store_true',
                    help='For causal true or false')

ZQ_parser.add_argument('--batch_size',
                    type=int,
                    default=16,
                    help='Minibatch size')

ZQ_parser.add_argument('--max_n_epochs',
                    type=int,
                    default=100,
                    help='Maximum number of epochs')

ZQ_parser.add_argument('--net_type',
                    type=str,
                    default='SISO',
                    help='MISO or MIMO or SISO'
                    )   
# distributed training
ZQ_parser.add_argument('--num_nodes',
                    type=int,
                    default=1,
                    help='Num of Nodes')
ZQ_parser.add_argument('--num_gpu_per_node',
                    type=int,
                    default=4,
                    help='Num GPU per Node')
ZQ_parser.add_argument('--num_workers',
                    type=int,
                    default=4,
                    help='Dataloader num_workers') 
# Network Initilaization and Resume model
ZQ_parser.add_argument('--pre_trained_ckpt_path',
                    type=str,
                    default='',
                    help='pre_trained_ckpt_path : Pytorch lightning checkpoint '
                    )
                    
ZQ_parser.add_argument('--resume_model',
                    type=str,
                    default='',
                    help='Existing model to resume training from')

ZQ_parser.add_argument('--model_path',
                    type=str,
                    default='',
                    help='Trained model to test')

ZQ_parser.add_argument('--nb_points',
                    type=int,
                    default=64,
                    help='nb points between start and end') 
#Testing Individual Jobs
ZQ_parser.add_argument('--test_snr',
                    type=int,
                    default=5,
                    help='Test SNR (dB)')               

ZQ_parser.add_argument('--test_t60',
                    type=float,
                    default=0.2,
                    help='Test T60 (sec)')
#Testing Array Jobs
ZQ_parser.add_argument('--input_test_filename',
                    type=str,
                    default='',
                    help='Absolute Test file path which contain Parameters required')

## Array Jobs flag

ZQ_parser.add_argument('--array_job',
                    action='store_true',
                    help='Trainong Array jobs model')

ZQ_parser.add_argument('--input_train_filename',
                    type=str,
                    default='',
                    help='Absolute Train file path which contain Parameters required')

if __name__=="__main__":
    args = ZQ_parser.parse_args()

