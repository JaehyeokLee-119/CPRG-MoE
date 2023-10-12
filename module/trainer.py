import logging
import os
import datetime
import tensorflow
import lightning.pytorch as L
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint

from module.lighttrainer import LitCPRGMoE
from module.preprocessing import get_data

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore TF error message 

class LearningEnv:
    def __init__(self, **kwargs):
        self.num_worker = kwargs['num_worker']
        self.model_save_path = kwargs['model_save_path']
        self.inference_model_path = kwargs['inference_model_path']
        self.context_type = kwargs['context_type']
        
        self.gpus = kwargs['gpus']
        self.single_gpu = len(self.gpus) == 1
        self.num_worker = kwargs['num_worker']

        self.dataset_type = kwargs['dataset_type']
        self.train_dataset = kwargs['train_data']
        self.valid_dataset = kwargs['valid_data']
        self.test_dataset = kwargs['test_data']
        self.data_label = kwargs['data_label']
        
        self.loss_lambda = kwargs['loss_lambda'] # loss 중 Emotion loss의 비율
        self.window_size = kwargs['window_size']
        
        self.start_time = datetime.datetime.now()
        self.training_iter = kwargs['training_iter']
        
        self.model_name = kwargs['model_name']
        self.port = kwargs['port']
        
        self.contain_context = kwargs['contain_context']
        self.max_seq_len = kwargs['max_seq_len']
        
        self.pretrained_model = kwargs['pretrained_model']
        # Hyperparameters
        self.dropout = kwargs['dropout']
        self.n_cause = kwargs['n_cause']
        self.n_speaker = kwargs['n_speaker']
        self.n_emotion = kwargs['n_emotion']
        self.n_expert = kwargs['n_expert']
        self.learning_rate = kwargs['learning_rate']
        self.batch_size = kwargs['batch_size']
        self.guiding_lambda = kwargs['guiding_lambda']
        self.encoder_name = kwargs['encoder_name']
        
        # learning variables
        self.best_performance = [0, 0, 0]  # p, r, f1
        self.num_epoch = 1
        self.accumulate_grad_batches = kwargs['accumulate_grad_batches']
        # set log directory
        self.encoder_name_for_filename = self.encoder_name.replace('/', '_')
        self.ckpt_filename = kwargs['ckpt_filename']
        
        # directory for saving logs
        self.log_directory = kwargs['log_directory']
            
        self.model_args = {
            "dropout": self.dropout,
            "n_speaker": self.n_speaker,
            "n_emotion": self.n_emotion,
            "n_cause": self.n_cause,
            "n_expert": self.n_expert,
            "guiding_lambda": self.guiding_lambda,
            "learning_rate": self.learning_rate,
            "loss_lambda": self.loss_lambda,
            "training_iter": self.training_iter,
            "encoder_name": self.encoder_name,
            "window_size": self.window_size,
            "contain_context": self.contain_context,
            "max_seq_len": self.max_seq_len,
            "dataset_type": self.dataset_type,
            "context_type": self.context_type,
        }

    def set_model(self):        
        if self.pretrained_model is not None:
            model = LitCPRGMoE.load_from_checkpoint(checkpoint_path=self.pretrained_model, **self.model_args)
        else:
            model = LitCPRGMoE(**self.model_args)
        self.model = model
    
    def run(self, **kwargs):
        self.pre_setting()
        if kwargs['test']:
            self.training_iter = 1
            self.inference()
        else:
            self.train()
            # self.test()
    
    def pre_setting(self):
        # 로거 설정
        logger_name_list = ['train', 'valid', 'test', 'samples']
        file_name_list = [f'{self.model_name}-{self.data_label}-{_}-{self.start_time}.log' for _ in logger_name_list]
        
        self.set_logger_environment(file_name_list, logger_name_list)
        
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        
        self.set_model()
    
    def train(self):
        train_dataloader = self.get_dataloader(self.train_dataset, self.dataset_type, self.batch_size, self.num_worker, shuffle=True, contain_context=self.contain_context, context_type=self.context_type)
        valid_dataloader = self.get_dataloader(self.valid_dataset, self.dataset_type, self.batch_size, self.num_worker, shuffle=False, contain_context=self.contain_context, context_type=self.context_type)
        test_dataloader = self.get_dataloader(self.test_dataset, self.dataset_type, self.batch_size, self.num_worker, shuffle=False, contain_context=self.contain_context, context_type=self.context_type)
        
        epoch = self.training_iter
        ckpt_filename = self.ckpt_filename
        model = LitCPRGMoE(**self.model_args)
        monitor_val = "emo-cau 3.f1-score"
            
        on_best_f1 = ModelCheckpoint(
            dirpath=self.model_save_path,
            monitor=monitor_val,
            save_top_k=1,
            mode="max",
            filename=ckpt_filename)
        
        trainer_config = {
            "max_epochs": epoch,
            "strategy": 'ddp_find_unused_parameters_true',
            "check_val_every_n_epoch": 1,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "callbacks": [on_best_f1]
        }
        trainer = L.Trainer(**trainer_config)
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
        
        # Test for Cause Performance
        model_path_cause = f'{self.model_save_path}/{ckpt_filename}.ckpt'
        
        test_dataloader = self.get_dataloader(self.test_dataset, self.dataset_type, 1, self.num_worker, shuffle=False, contain_context=self.contain_context, context_type=self.context_type)
        model = LitCPRGMoE.load_from_checkpoint(checkpoint_path=model_path_cause, **self.model_args)
        trainer.test(model, dataloaders=test_dataloader)
    
    def inference(self):
        test_dataloader = self.get_dataloader(self.test_dataset, self.dataset_type, self.batch_size, self.num_worker, shuffle=False, contain_context=self.contain_context, context_type=self.context_type)
        
        model_path = self.inference_model_path
        self.model = LitCPRGMoE.load_from_checkpoint(checkpoint_path=model_path, **self.model_args)
        trainer = L.Trainer()
        trainer.test(self.model, dataloaders=test_dataloader)  
    
    def test(self):
        test_dataloader = self.get_dataloader(self.test_dataset, self.dataset_type, self.batch_size, self.num_worker, shuffle=False, contain_context=self.contain_context, context_type=self.context_type)
        
        self.model_args['batch_size'] = 1
        self.model = LitCPRGMoE.load_from_checkpoint(checkpoint_path=self.pretrained_model, **self.model_args)
        trainer = L.Trainer()
        trainer.test(self.model, dataloaders=test_dataloader)  
    
    def set_logger_environment(self, file_name_list, logger_name_list):
        for file_name, logger_name in zip(file_name_list, logger_name_list):
            for handler in logging.getLogger(logger_name).handlers[:]:
                logging.getLogger(logger_name).removeHandler(handler)
            self.set_logger(file_name, logger_name)

    def set_logger(self, file_name, logger_name):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if self.log_directory:
            if not os.path.exists(f'{self.log_directory}'):
                os.makedirs(f'{self.log_directory}')
            file_handler = logging.FileHandler(f'{self.log_directory}/{file_name}')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def get_dataloader(self, dataset_type, dataset_file, batch_size, num_worker, shuffle=True, contain_context=False, context_type='xxx'):
        device = "cuda:0"
        # dataset_type = ["ConvECPE", "RECCON"]
        data = get_data(dataset_type, dataset_file, device, self.max_seq_len, self.encoder_name, contain_context, context_type=self.context_type)
        utterance_input_ids_t, utterance_attention_mask_t, utterance_token_type_ids_t = data[0]
        speaker_t, emotion_label_t, pair_cause_label_t, pair_binary_cause_label_t = data[1:]

        dataset_ = TensorDataset(utterance_input_ids_t, utterance_attention_mask_t, utterance_token_type_ids_t, speaker_t, emotion_label_t, pair_cause_label_t, pair_binary_cause_label_t)
        
        dataloader_params = {
            "dataset": dataset_,
            "batch_size": batch_size,
            "num_workers": num_worker,
            # "shuffle": shuffle
        }
        
        return DataLoader(**dataloader_params)