import argparse
import os
import random
from typing import List

import numpy as np
import torch

from datetime import datetime

from module.trainer import LearningEnv
from dotenv import load_dotenv

def set_random_seed(seed: int):
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='This code is for ECPEC task.')

    # Training Environment
    parser.add_argument('--gpus', default="3")
    parser.add_argument('--num_process', default=int(os.cpu_count() * 0.8), type=int)
    parser.add_argument('--num_worker', default=6, type=int)
    parser.add_argument('--port', default=1234, type=int)

    parser.add_argument('--model_name', default='CPRG_MoE')
    parser.add_argument('--pretrained_model', default=None)
    parser.add_argument('--inference_model_path', help='path of model checkpoint for test', default=None)
    parser.add_argument('--test', default=False)
    parser.add_argument('--model_save_path', default='./model')

    parser.add_argument('--train_data', default="data/data_ConvECPE/data_0/ConvECPE_fold_0_train.json")
    parser.add_argument('--valid_data', default="data/data_ConvECPE/data_0/ConvECPE_fold_0_valid.json")
    parser.add_argument('--test_data', default="data/data_ConvECPE/data_0/ConvECPE_fold_0_test.json")
    parser.add_argument('--dataset_type', help='RECCON or ConvECPE', default='ConvECPE')
    parser.add_argument('--data_label', help='the label that attaches to the name of saved model', default='ConvECPE_0')
    parser.add_argument('--n_emotion', help='the number of emotions', default=6, type=int)
    parser.add_argument('--log_directory', default='logs', type=str)
    parser.add_argument('--ckpt_filename', help='file name of model checkpoint to be saved', default='ckpt', type=str)

    # Encoder Model Setting
    parser.add_argument('--encoder_name', help='the name of encoder', default='bert-base-cased')
    parser.add_argument('--max_seq_len', help='the max length of each tokenized utterance', default=128, type=int)
    parser.add_argument('--loss_lambda', help='Ratio of emotion loss in the total loss', default=0.2)
    
    # Training Hyperparameters
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--guiding_lambda', help='the mixing ratio', default=0.6, type=float)
    parser.add_argument('--training_iter', default=15, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--patience', help='patience for Early Stopping', default=None, type=int)
    parser.add_argument('--accumulate_grad_batches', default=1, type=int)
    parser.add_argument('--window_size', default=3, type=int)
    parser.add_argument('--contain_context', help='Incorporate contextual information', default=True)
    parser.add_argument('--context_type', help='context modeling approach: 1: SEP, 2: Rev, 3: Spk', default='xvd', type=str)
    
    parser.add_argument('--n_speaker', help='the number of speakers', default=2, type=int)
    parser.add_argument('--n_cause', help='the number of causes', default=2, type=int)
    parser.add_argument('--n_expert', help='the number of causes', default=4, type=int)
    return parser.parse_args()


def test_preconditions(args: argparse.Namespace):
    if args.test:
        assert args.pretrained_model is not None, "For test, you should load pretrained model."


class Main:
    def __init__(self):
        load_dotenv()
        self.args = parse_args()
        test_preconditions(self.args)
        set_random_seed(77)
        
    def run(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(_) for _ in self.args.gpus])
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore TensorFlow error message 
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ["WANDB_DISABLED"] = "true"
        
        if self.args.dataset_type == 'ConvECPE':
            self.args.n_emotion = 6
        else:
            self.args.n_emotion = 7
        
        if type(self.args.gpus) == str: 
            self.args.gpus = self.args.gpus.split(',')
            self.args.gpus = [int(_) for _ in self.args.gpus]
                    
                    
        trainer = LearningEnv(**vars(self.args))
        trainer.run(**vars(self.args))
        del trainer
        
        torch.cuda.empty_cache()
           
if __name__ == "__main__":
    main = Main()
    main.run()
    
