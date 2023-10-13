import lightning as pl
import torch
import logging
from module.evaluation import FocalLoss
from module.preprocessing import get_pair_pad_idx, get_pad_idx
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
from module.lightmodels import CPRG_MoE

class LitCPRGMoE(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.encoder_name = kwargs['encoder_name']
        self.n_expert = kwargs['n_expert']
        self.dataset_type = kwargs['dataset_type']
        self.n_emotion = kwargs['n_emotion']
        self.guiding_lambda = kwargs['guiding_lambda']
        
        if self.n_emotion == 7:
            self.label_neutral = 6
        else:
            self.label_neutral = 2
            
        self.model = CPRG_MoE(self.encoder_name, guiding_lambda=self.guiding_lambda , n_emotion=self.n_emotion) # output: (emotion prediction, cause prediction)

        self.training_iter = kwargs['training_iter']
        self.dropout = kwargs['dropout']
        self.learning_rate = kwargs['learning_rate']
        self.window_size = kwargs['window_size']
        self.loss_lambda = kwargs['loss_lambda']
        self.n_cause = kwargs['n_cause']
        
        self.test = False # True when testing(on_test_epoch_start ~ on_test_epoch_end)
        
        if 'bert-base' in self.encoder_name:
            self.is_bert_like = True
        else:
            self.is_bert_like = False
                        
        # Dictionaries for logging
        types = ['train', 'valid', 'test']
        
        self.emo_pred_y_list = {}
        self.emo_true_y_list = {}
        self.cau_pred_y_list = {}
        self.cau_true_y_list = {}
        self.cau_pred_y_list_all = {}
        self.cau_true_y_list_all = {}
        self.cau_pred_y_list_all_windowed = {}
        self.emo_cause_pred_y_list = {}
        self.emo_cause_true_y_list = {}
        self.emo_cause_pred_y_list_all = {}
        self.emo_cause_true_y_list_all = {}
        self.pair_label_indegree_batch_window = {}
        self.pair_label_indegree_batch_all = {}
        self.loss_sum = {}
        self.batch_count = {}
        
        for i in types:
            self.emo_pred_y_list[i] = []
            self.emo_true_y_list[i] = []
            self.cau_pred_y_list[i] = []
            self.cau_true_y_list[i] = []
            self.cau_pred_y_list_all[i] = []
            self.cau_true_y_list_all[i] = []
            self.cau_pred_y_list_all_windowed[i] = []
            self.emo_cause_pred_y_list[i] = []
            self.emo_cause_true_y_list[i] = []
            self.emo_cause_pred_y_list_all[i] = []
            self.emo_cause_true_y_list_all[i] = []
            self.pair_label_indegree_batch_window[i] = []
            self.pair_label_indegree_batch_all[i] = []
            self.loss_sum[i] = 0.0
            self.batch_count[i] = 0
        
        self.best_performance_emo = {
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'weighted_f1': 0.0,
            'epoch': 0,
            'loss': 0.0,
        }
        self.best_performance_cau = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'epoch': 0,
            'loss': 0.0,
        }
        self.best_performance_emo_cau = {
            'epoch': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
        }
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=5,
                                                    num_training_steps=self.training_iter,
                                                    )
        return [optimizer], [scheduler]
                    
    def forward(self, batch):
        utterance_input_ids_batch, utterance_attention_mask_batch, utterance_token_type_ids_batch, speaker_batch, emotion_label_batch, pair_cause_label_batch, pair_binary_cause_label_batch = batch
        
        batch_size, max_doc_len, max_seq_len = utterance_input_ids_batch.shape
        
        input_ids = utterance_input_ids_batch
        attention_mask = utterance_attention_mask_batch
        token_type_ids = utterance_token_type_ids_batch
        speaker_ids = speaker_batch
        
        # Forward
        emotion_prediction, cause_prediction = self.model(input_ids, attention_mask, token_type_ids, speaker_ids, max_seq_len)
        
        return (emotion_prediction, cause_prediction)
    
    def output_processing(self, utterance_input_ids_batch, pair_binary_cause_label_batch, emotion_label_batch, emotion_prediction, binary_cause_prediction):
        batch_size, dialog_len, _ = utterance_input_ids_batch.shape
        check_pair_window_idx = get_pair_pad_idx(utterance_input_ids_batch, self.encoder_name, window_constraint=self.window_size, emotion_pred=emotion_prediction, label_neutral=self.label_neutral)
        check_pair_pad_idx = get_pair_pad_idx(utterance_input_ids_batch, self.encoder_name, window_constraint=1000, label_neutral=self.label_neutral, )
        check_pad_idx = get_pad_idx(utterance_input_ids_batch, self.encoder_name)

        emotion_list = emotion_prediction.view(batch_size, -1, self.n_emotion)
        emotion_pair_list = []
        emotion_pred_list = []
        for doc_emotion in emotion_list:
                end_t = 0
                for utt_emotion in doc_emotion:
                    emotion_pred_list.append(torch.argmax(utt_emotion))
                    for _ in range(end_t+1):
                        emotion_pair_list.append(torch.argmax(utt_emotion)) 
                    end_t += 1
        emotion_label_pair_list = [] 
        for doc_emotion in emotion_label_batch:
            end_t = 0
            for emotion in doc_emotion:
                for _ in range(end_t+1):
                    emotion_label_pair_list.append(emotion)
                end_t += 1
        emotion_pair_pred_expanded = torch.stack(emotion_pair_list).view(batch_size, -1)
        emotion_pair_true_expanded = torch.stack(emotion_label_pair_list).view(batch_size, -1)
        
        pair_binary_cause_label_indegree_batch = torch.zeros_like(pair_binary_cause_label_batch)
        a = 0
        b = 0
        for i in range(1, dialog_len+1):
            b += i
            for j in range(batch_size):
                pair_binary_cause_label_indegree_batch[j][a:b] = pair_binary_cause_label_batch[j][a:b].count_nonzero()
            a = b
    
        pair_binary_cause_label_indegree_batch_window = pair_binary_cause_label_indegree_batch[(check_pair_window_idx != False).nonzero(as_tuple=True)]
        pair_binary_cause_label_indegree_batch_all = pair_binary_cause_label_indegree_batch[(check_pair_pad_idx != False).nonzero(as_tuple=True)]
        
        emotion_prediction_filtered = emotion_prediction[(check_pad_idx != False).nonzero(as_tuple=True)]
        emotion_label_batch_filtered = emotion_label_batch.view(-1)[(check_pad_idx != False).nonzero(as_tuple=True)]
        
        pair_binary_cause_prediction_window = binary_cause_prediction.view(batch_size, -1, self.n_cause)[(check_pair_window_idx != False).nonzero(as_tuple=True)]
        
        pair_prediction_argmax = torch.argmax(binary_cause_prediction, dim=1)
        pair_prediction_argmax_windowed = pair_prediction_argmax.cpu() * check_pair_window_idx.view(-1)
        pair_prediction_argmax_windowed_all = pair_prediction_argmax_windowed.view(batch_size, -1)[(check_pair_pad_idx != False).nonzero(as_tuple=True)]
        
        pair_binary_cause_prediction_all = binary_cause_prediction.view(batch_size, -1, self.n_cause)[(check_pair_pad_idx != False).nonzero(as_tuple=True)]
        
        pair_binary_cause_label_batch_window = pair_binary_cause_label_batch[(check_pair_window_idx != False).nonzero(as_tuple=True)]
        pair_binary_cause_label_batch_all = pair_binary_cause_label_batch[(check_pair_pad_idx != False).nonzero(as_tuple=True)]
        
        pair_emotion_prediction_window = emotion_pair_pred_expanded[(check_pair_window_idx != False).nonzero(as_tuple=True)]
        pair_emotion_prediction_all = emotion_pair_pred_expanded[(check_pair_pad_idx != False).nonzero(as_tuple=True)]
        pair_emotion_label_batch_window = emotion_pair_true_expanded[(check_pair_window_idx != False).nonzero(as_tuple=True)]
        pair_emotion_label_batch_all = emotion_pair_true_expanded[(check_pair_pad_idx != False).nonzero(as_tuple=True)]
        
        pair_emotion_cause_prediction_window = pair_emotion_prediction_window*10 + torch.argmax(pair_binary_cause_prediction_window, dim=1)
        pair_emotion_cause_prediction_all = pair_emotion_prediction_all*10 + torch.argmax(pair_binary_cause_prediction_all, dim=1)
        pair_emotion_cause_label_batch_window = pair_emotion_label_batch_window*10 + pair_binary_cause_label_batch_window
        pair_emotion_cause_label_batch_all = pair_emotion_label_batch_all*10 + pair_binary_cause_label_batch_all
        
        emotion_ = (emotion_prediction_filtered, emotion_label_batch_filtered)
        cause_ = (pair_binary_cause_prediction_window, pair_binary_cause_prediction_all, pair_binary_cause_label_batch_window, pair_binary_cause_label_batch_all, pair_prediction_argmax_windowed_all)
        emotion_cause_ = (pair_emotion_cause_prediction_window, pair_emotion_cause_prediction_all, pair_emotion_cause_label_batch_window, pair_emotion_cause_label_batch_all)
        cause_degrees_ = (pair_binary_cause_label_indegree_batch_window, pair_binary_cause_label_indegree_batch_all)
        return (emotion_, cause_, emotion_cause_, cause_degrees_)
    
    
    
    def loss_calculation(self, emotion_prediction_filtered, emotion_label_batch_filtered, pair_binary_cause_prediction_window, pair_binary_cause_label_batch_window):
        criterion_emo = FocalLoss(gamma=2)
        criterion_cau = FocalLoss(gamma=2)
        
        loss_emo = criterion_emo(emotion_prediction_filtered, emotion_label_batch_filtered)
        loss_cau = criterion_cau(pair_binary_cause_prediction_window, pair_binary_cause_label_batch_window)
        loss = self.loss_lambda * loss_emo + (1-self.loss_lambda) * loss_cau
        return loss
    
    
    def training_step(self, batch, batch_idx):
        types = 'train'
        utterance_input_ids_batch, _, _, _, emotion_label_batch, _, pair_binary_cause_label_batch = batch
        
        emotion_prediction, binary_cause_prediction = self.forward(batch)
        
        (emotion_, cause_, emotion_cause_, cause_degrees_) = self.output_processing(utterance_input_ids_batch, pair_binary_cause_label_batch, emotion_label_batch, emotion_prediction, binary_cause_prediction)
        
        (emotion_prediction_filtered, emotion_label_batch_filtered) = emotion_
        (pair_binary_cause_prediction_window, pair_binary_cause_prediction_all, pair_binary_cause_label_batch_window, pair_binary_cause_label_batch_all, pair_prediction_argmax_windowed_all) = cause_
        (pair_emotion_cause_prediction_window, pair_emotion_cause_prediction_all, pair_emotion_cause_label_batch_window, pair_emotion_cause_label_batch_all) = emotion_cause_
        (pair_binary_cause_label_indegree_batch_window, pair_binary_cause_label_indegree_batch_all) = cause_degrees_
        
        self.cau_pred_y_list_all[types].append(pair_binary_cause_prediction_all), self.cau_true_y_list_all[types].append(pair_binary_cause_label_batch_all)
        self.cau_pred_y_list[types].append(pair_binary_cause_prediction_window), self.cau_true_y_list[types].append(pair_binary_cause_label_batch_window)
        self.cau_pred_y_list_all_windowed[types].append(pair_prediction_argmax_windowed_all)
        self.emo_pred_y_list[types].append(emotion_prediction_filtered), self.emo_true_y_list[types].append(emotion_label_batch_filtered)
        self.emo_cause_pred_y_list[types].append(pair_emotion_cause_prediction_window)
        self.emo_cause_true_y_list[types].append(pair_emotion_cause_label_batch_window)
        self.emo_cause_pred_y_list_all[types].append(pair_emotion_cause_prediction_all)
        self.emo_cause_true_y_list_all[types].append(pair_emotion_cause_label_batch_all)
        self.pair_label_indegree_batch_window[types].append(pair_binary_cause_label_indegree_batch_window)
        self.pair_label_indegree_batch_all[types].append(pair_binary_cause_label_indegree_batch_all)
        
        loss = self.loss_calculation(emotion_prediction_filtered, emotion_label_batch_filtered, pair_binary_cause_prediction_window, pair_binary_cause_label_batch_window)
        
        self.loss_sum[types] += loss.item()
        self.batch_count[types] += 1
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        types = 'valid'
        utterance_input_ids_batch, _, _, _, emotion_label_batch, _, pair_binary_cause_label_batch = batch
        
        emotion_prediction, binary_cause_prediction = self.forward(batch)
        
        (emotion_, cause_, emotion_cause_, cause_degrees_) = self.output_processing(utterance_input_ids_batch, pair_binary_cause_label_batch, emotion_label_batch, emotion_prediction, binary_cause_prediction)
        
        (emotion_prediction_filtered, emotion_label_batch_filtered) = emotion_
        (pair_binary_cause_prediction_window, pair_binary_cause_prediction_all, pair_binary_cause_label_batch_window, pair_binary_cause_label_batch_all, pair_prediction_argmax_windowed_all) = cause_
        (pair_emotion_cause_prediction_window, pair_emotion_cause_prediction_all, pair_emotion_cause_label_batch_window, pair_emotion_cause_label_batch_all) = emotion_cause_
        (pair_binary_cause_label_indegree_batch_window, pair_binary_cause_label_indegree_batch_all) = cause_degrees_
        
        loss = self.loss_calculation(emotion_prediction_filtered, emotion_label_batch_filtered, pair_binary_cause_prediction_window, pair_binary_cause_label_batch_window)
            
        self.log("valid_loss: ", loss, sync_dist=True)
        self.cau_pred_y_list_all[types].append(pair_binary_cause_prediction_all), self.cau_true_y_list_all[types].append(pair_binary_cause_label_batch_all)
        self.cau_pred_y_list[types].append(pair_binary_cause_prediction_window), self.cau_true_y_list[types].append(pair_binary_cause_label_batch_window)
        self.cau_pred_y_list_all_windowed[types].append(pair_prediction_argmax_windowed_all)
        self.emo_pred_y_list[types].append(emotion_prediction_filtered), self.emo_true_y_list[types].append(emotion_label_batch_filtered)
        self.emo_cause_pred_y_list[types].append(pair_emotion_cause_prediction_window)
        self.emo_cause_true_y_list[types].append(pair_emotion_cause_label_batch_window)
        self.emo_cause_pred_y_list_all[types].append(pair_emotion_cause_prediction_all)
        self.emo_cause_true_y_list_all[types].append(pair_emotion_cause_label_batch_all)
        self.pair_label_indegree_batch_window[types].append(pair_binary_cause_label_indegree_batch_window)
        self.pair_label_indegree_batch_all[types].append(pair_binary_cause_label_indegree_batch_all)
        
        self.loss_sum[types] += loss.item()
        self.batch_count[types] += 1
        
    def test_step(self, batch, batch_idx):
        types = 'test'
        utterance_input_ids_batch, utterance_attention_mask_batch, utterance_token_type_ids_batch, speaker_batch, emotion_label_batch, pair_cause_label_batch, pair_binary_cause_label_batch = batch
        batch_size, dialog_len, _ = utterance_input_ids_batch.shape
        
        emotion_prediction, binary_cause_prediction = self.forward(batch)
        
        (emotion_, cause_, emotion_cause_, cause_degrees_) = self.output_processing(utterance_input_ids_batch, pair_binary_cause_label_batch, emotion_label_batch, emotion_prediction, binary_cause_prediction)
        
        (emotion_prediction_filtered, emotion_label_batch_filtered) = emotion_
        (pair_binary_cause_prediction_window, pair_binary_cause_prediction_all, pair_binary_cause_label_batch_window, pair_binary_cause_label_batch_all, pair_prediction_argmax_windowed_all) = cause_
        (pair_emotion_cause_prediction_window, pair_emotion_cause_prediction_all, pair_emotion_cause_label_batch_window, pair_emotion_cause_label_batch_all) = emotion_cause_
        (pair_binary_cause_label_indegree_batch_window, pair_binary_cause_label_indegree_batch_all) = cause_degrees_
        
        check_pad_idx = get_pad_idx(utterance_input_ids_batch, self.encoder_name)
        utterance_filtered = utterance_input_ids_batch.view(dialog_len, -1).cpu()[(check_pad_idx != False).nonzero(as_tuple=True)]
        
        dialog_len_filtered = utterance_filtered.shape[0]
        tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        utterance_filtered_string_list = []
        for i in range(dialog_len_filtered):
            cleared_token_sequence = utterance_filtered[i][utterance_filtered[i] != 0][1:(utterance_filtered[i][utterance_filtered[i] != 0] == 102).nonzero(as_tuple=True)[0][0]]
            utterance_filtered_string_list.append(tokenizer.decode(cleared_token_sequence))
        
        if self.dataset_type == 'RECCON':
            policy_to_array = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        else:
            policy_to_array = ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated']
        emotion_prediction_list = []
        for i in range(dialog_len_filtered):
            pred_emotion_num = torch.argmax(emotion_prediction_filtered, dim=1)[i]
            emotion_prediction_list.append(policy_to_array[pred_emotion_num])
        
        emotion_label_list = []
        for i in range(dialog_len_filtered):
            emotion_label_list.append(policy_to_array[emotion_label_batch_filtered[i]])
            
        pair_prediction_list = []
        for i in range(1, dialog_len_filtered+1):
            for j in range(i):
                emo_index = int(i*(i+1)/2) - 1
                cau_index = int(i*(i-1)/2) + j
                if (pair_prediction_argmax_windowed_all[cau_index] == 1):
                    pair_prediction_list.append((i, j+1))
        
        pair_label_list = []
        for i in range(1, dialog_len_filtered+1):
            pair_binary_cause_label_batch_all
            for j in range(i):
                emo_index = int(i*(i+1)/2) - 1
                cau_index = int(i*(i-1)/2) + j
                if (pair_binary_cause_label_batch_all[cau_index] == 1):
                    pair_label_list.append((i, j+1))
                    
        logger = logging.getLogger("samples")
        logged_text = ''
        logged_text += '<Start>\nPred: '
        for pair_pred in pair_prediction_list:
            if pair_pred != pair_prediction_list[-1]:
                logged_text += f'{pair_pred}, '
            else: 
                logged_text += f'{pair_pred}'
        logged_text += '\nTrue: '
        for pair_true in pair_label_list:
            if pair_true != pair_label_list[-1]:
                logged_text += f'{pair_true}, '
            else:
                logged_text += f'{pair_true} '
        logged_text += '\n'
        for i, (text, emo, emo_label) in enumerate(zip(utterance_filtered_string_list, emotion_prediction_list, emotion_label_list)):
            logged_text += f'[{i+1}] {emo}({emo_label}):{text}\n'
        logged_text += '<End>\n'
        logger.info(logged_text)
        
        # Loss Calculation
        loss = self.loss_calculation(emotion_prediction_filtered, emotion_label_batch_filtered, pair_binary_cause_prediction_window, pair_binary_cause_label_batch_window)
            
        self.log("test_loss: ", loss, sync_dist=True)
        # Logging
        self.cau_pred_y_list_all[types].append(pair_binary_cause_prediction_all), self.cau_true_y_list_all[types].append(pair_binary_cause_label_batch_all)
        self.cau_pred_y_list[types].append(pair_binary_cause_prediction_window), self.cau_true_y_list[types].append(pair_binary_cause_label_batch_window)
        self.cau_pred_y_list_all_windowed[types].append(pair_prediction_argmax_windowed_all)
        self.emo_pred_y_list[types].append(emotion_prediction_filtered), self.emo_true_y_list[types].append(emotion_label_batch_filtered)
        self.emo_cause_pred_y_list[types].append(pair_emotion_cause_prediction_window)
        self.emo_cause_true_y_list[types].append(pair_emotion_cause_label_batch_window)
        self.emo_cause_pred_y_list_all[types].append(pair_emotion_cause_prediction_all)
        self.emo_cause_true_y_list_all[types].append(pair_emotion_cause_label_batch_all)
        self.pair_label_indegree_batch_window[types].append(pair_binary_cause_label_indegree_batch_window)
        self.pair_label_indegree_batch_all[types].append(pair_binary_cause_label_indegree_batch_all)
        
        self.loss_sum[types] += loss.item()
        self.batch_count[types] += 1

    def on_train_epoch_start(self):
        self.make_test_setting(types='train')
        
    def on_train_epoch_end(self):
        self.log_test_result(types='train')
    
    def on_validation_epoch_start(self):
        self.test = True
        self.make_test_setting(types='valid')

    def on_validation_epoch_end(self):
        self.test = False
        self.log_test_result(types='valid')
    
    def on_test_epoch_start(self):
        self.test = True
        self.make_test_setting(types='test')
        
    def on_test_epoch_end(self):
        self.test = False
        self.log_test_result(types='test')
        
    def make_test_setting(self, types='train'):
        self.emo_pred_y_list[types] = []
        self.emo_true_y_list[types] = []
        self.cau_pred_y_list[types] = []
        self.cau_true_y_list[types] = []
        self.cau_pred_y_list_all[types] = []
        self.cau_true_y_list_all[types] = []
        self.cau_pred_y_list_all_windowed[types] = []
        self.emo_cause_pred_y_list[types] = []
        self.emo_cause_true_y_list[types] = []
        self.emo_cause_pred_y_list_all[types] = []
        self.emo_cause_true_y_list_all[types] = []
        self.pair_label_indegree_batch_window[types] = []
        self.pair_label_indegree_batch_all[types] = []
        self.loss_sum[types] = 0.0
        self.batch_count[types] = 0
        
    def log_test_result(self, types='train'):
        logger = logging.getLogger(types)
        
        loss_avg = self.loss_sum[types] / self.batch_count[types]
        emo_report, emo_metrics, emo_binary_report, additional_report, acc_cau, p_cau, r_cau, f1_cau, p_emo_cau, r_emo_cau, f1_emo_cau = log_metrics(self.dataset_type, self.emo_pred_y_list[types], self.emo_true_y_list[types], 
                                                self.cau_pred_y_list[types], self.cau_true_y_list[types],
                                                self.cau_pred_y_list_all[types], self.cau_true_y_list_all[types], 
                                                self.cau_pred_y_list_all_windowed[types],
                                                self.emo_cause_pred_y_list[types], self.emo_cause_true_y_list[types],
                                                self.emo_cause_pred_y_list_all[types], self.emo_cause_true_y_list_all[types],
                                                self.pair_label_indegree_batch_window[types], self.pair_label_indegree_batch_all[types],
                                                loss_avg)
        
        self.log('binary_cause 1.loss', loss_avg, sync_dist=True)
        self.log('binary_cause 2.accuracy', acc_cau, sync_dist=True)
        self.log('binary_cause 3.precision', p_cau, sync_dist=True)
        self.log('binary_cause 4.recall', r_cau, sync_dist=True)
        self.log('binary_cause 5.f1-score', f1_cau, sync_dist=True)
        
        self.log('emo 1.accuracy', emo_metrics[0], sync_dist=True)
        self.log('emo 2.macro-f1', emo_metrics[1], sync_dist=True)
        self.log('emo 3.weighted-f1', emo_metrics[2], sync_dist=True)
        
        self.log('emo-cau 1.precision', p_emo_cau, sync_dist=True)
        self.log('emo-cau 2.recall', r_emo_cau, sync_dist=True)
        self.log('emo-cau 3.f1-score', f1_emo_cau, sync_dist=True)
        
        logging_texts = f'\n[Epoch {self.current_epoch}] / <Emotion Prediction> of {types}\n'+\
                        f'\n<Emotion Multiclass Classfication Performance>\n'+\
                        emo_report+\
                        f'\n<Emotion Binary Classfication Performance>\n'+\
                        emo_binary_report+\
                        f'\n<ECPD>'+\
                        f'\n\tprecision:\t     {p_cau}'+\
                        f'\n\trecall:   \t     {r_cau}'+\
                        f'\n\tf1-score: \t     {f1_cau}'+\
                        f'\n\n<PECA>'+\
                        f'\n\tprecision:\t     {p_emo_cau}'+\
                        f'\n\trecall:   \t     {r_emo_cau}'+\
                        f'\n\tf1-score: \t     {f1_emo_cau}'+\
                        f'\n'+\
                        additional_report
                        
        if (types == 'valid'):
            if (self.best_performance_emo['weighted_f1'] < emo_metrics[2]):
                self.best_performance_emo['weighted_f1'] = emo_metrics[2]
                self.best_performance_emo['accuracy'] = emo_metrics[0]
                self.best_performance_emo['macro_f1'] = emo_metrics[1]
                self.best_performance_emo['epoch'] = self.current_epoch
                self.best_performance_emo['loss'] = loss_avg
            if (self.best_performance_cau['f1'] < f1_cau):
                self.best_performance_cau['f1'] = f1_cau
                self.best_performance_cau['accuracy'] = acc_cau
                self.best_performance_cau['precision'] = p_cau
                self.best_performance_cau['recall'] = r_cau
                self.best_performance_cau['epoch'] = self.current_epoch
                self.best_performance_cau['loss'] = loss_avg
            if (self.best_performance_emo_cau['f1'] < f1_emo_cau):
                self.best_performance_emo_cau['precision'] = p_emo_cau
                self.best_performance_emo_cau['recall'] = r_emo_cau
                self.best_performance_emo_cau['f1'] = f1_emo_cau
                self.best_performance_emo_cau['epoch'] = self.current_epoch
            
            appended_log_valid = f'\nCurrent Best Performance: loss: {self.best_performance_cau["loss"]}\n'+\
                            f'\t<Emotion Prediction: [Epoch: {self.best_performance_emo["epoch"]}]>\n'+\
                            f'\t\taccuracy: \t{self.best_performance_emo["accuracy"]}\n'+\
                            f'\t\tmacro_f1: \t{self.best_performance_emo["macro_f1"]}\n'+\
                            f'\t\tweighted_f1: \t{self.best_performance_emo["weighted_f1"]}\n'+\
                            f'\t<Emotion-Cause Prediction: [Epoch: {self.best_performance_cau["epoch"]}]>\n'+\
                            f'\t\taccuracy: \t{self.best_performance_cau["accuracy"]}\n'+\
                            f'\t\tprecision: \t{self.best_performance_cau["precision"]}\n'+\
                            f'\t\trecall: \t{self.best_performance_cau["recall"]}\n'+\
                            f'\t\tf1:\t\t{self.best_performance_cau["f1"]}\n'+\
                            f'\t<PECA: [Epoch: {self.best_performance_emo_cau["epoch"]}]>\n'+\
                            f'\t\tprecision: \t{self.best_performance_emo_cau["precision"]}\n'+\
                            f'\t\trecall: \t{self.best_performance_emo_cau["recall"]}\n'+\
                            f'\t\tf1:\t\t{self.best_performance_emo_cau["f1"]}\n'
            
        if (types == 'valid'):
            logging_texts += appended_log_valid
            
        logger.info(logging_texts)
        if (types == 'test'):            
            if self.dataset_type == 'RECCON':
                label_ = np.array([1, 11, 21, 31, 41, 51, 61, 0, 10, 20, 30, 40, 50, 60])
            elif self.dataset_type == 'ConvECPE':
                label_ = np.array([1, 11, 21, 31, 41, 51, 0, 10, 20, 30, 40, 50])
            confusion_pred = confusion_matrix(torch.cat(self.emo_cause_true_y_list[types]).to('cpu'), torch.cat(self.emo_cause_pred_y_list[types]).to('cpu'), labels=label_)
            confusion_all = confusion_matrix(torch.cat(self.emo_cause_true_y_list_all[types]).to('cpu'), torch.cat(self.emo_cause_pred_y_list_all[types]).to('cpu'), labels=label_)
            
            confusion_log = ""
            confusion_log+="[Confusion_pred matrix]\n"
            confusion_log+='\t'
            for label in label_:
                confusion_log+=f'{label}\t'
            confusion_log+="\n"
            for row, label in zip(confusion_pred, label_):
                confusion_log+=f'{label}\t'
                for col in row:
                    confusion_log+=f'{col}\t'
                confusion_log+="\n"
            confusion_log+="[Confusion_all matrix]\n"
            confusion_log+='\t'
            for label in label_:
                confusion_log+=f'{label}\t'
            confusion_log+="\n"
            for row, label in zip(confusion_all, label_):
                confusion_log+=f'{label}\t'
                for col in row:
                    confusion_log+=f'{col}\t'
                confusion_log+="\n"
            logger.info(confusion_log)
            # logger.info(confusion_log)
            
        
    def get_pair_embedding(self, pooled_output, emotion_prediction, input_ids, attention_mask, token_type_ids, speaker_ids):
        batch_size, max_doc_len, max_seq_len = input_ids.shape
        
        utterance_representation = self.dropout(pooled_output)

        concatenated_embedding = torch.cat((utterance_representation, emotion_prediction, 
                                            speaker_ids.view(-1).unsqueeze(1)), dim=1)
        
        pair_embedding = list()
        for batch in concatenated_embedding.view(batch_size, max_doc_len, -1):
            pair_per_batch = list()
            for end_t in range(max_doc_len):
                for t in range(end_t + 1):
                    pair_per_batch.append(torch.cat((batch[t], batch[end_t])))
            pair_embedding.append(torch.stack(pair_per_batch))

        pair_embedding = torch.stack(pair_embedding).to(input_ids.device)

        return pair_embedding
    
    def get_subtask_label(self, input_ids, speaker_ids, emotion_prediction):
        batch_size, max_doc_len, max_seq_len = input_ids.shape

        pair_info = []
        for speaker_batch, emotion_batch in zip(speaker_ids.view(batch_size, max_doc_len, -1), emotion_prediction.view(batch_size, max_doc_len, -1)):
            info_pair_per_batch = []
            for end_t in range(max_doc_len):
                for t in range(end_t + 1):
                    speaker_condition = speaker_batch[t] == speaker_batch[end_t]
                    emotion_condition = torch.argmax(
                        emotion_batch[t]) == torch.argmax(emotion_batch[end_t])

                    if speaker_condition and emotion_condition:
                        # if speaker and dominant emotion are same
                        info_pair_per_batch.append(torch.Tensor([1, 0, 0, 0]))
                    elif speaker_condition:
                        # if speaker is same, but dominant emotion is differnt
                        info_pair_per_batch.append(torch.Tensor([0, 1, 0, 0]))
                    elif emotion_condition:
                        # if speaker is differnt, but dominant emotion is same
                        info_pair_per_batch.append(torch.Tensor([0, 0, 1, 0]))
                    else:
                        # if speaker and dominant emotion are differnt
                        info_pair_per_batch.append(torch.Tensor([0, 0, 0, 1]))
            pair_info.append(torch.stack(info_pair_per_batch))

        pair_info = torch.stack(pair_info).to(input_ids.device)

        return pair_info
    
def metrics_report_for_emo_binary(pred_y, true_y, label_neutral=6, get_dict=False, multilabel=False):
    if multilabel:
        pred_y, true_y = threshold_prediction(pred_y, true_y)
        available_label = sorted(list(set((pred_y == True).nonzero()[:, -1].tolist() + (true_y == True).nonzero()[:, -1].tolist())))
    else:
        pred_y, true_y = argmax_prediction(pred_y, true_y)
        available_label = sorted(list(set(true_y.tolist() + pred_y.tolist())))

    class_name = ['non-neutral', 'neutral']
    
    pred_y = [1 if element == label_neutral else 0 for element in pred_y]
    true_y = [1 if element == label_neutral else 0 for element in true_y]

    if get_dict:
        return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4, output_dict=True)
    else:
        return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4)

    
def log_metrics(dataset_type, emo_pred_y_list, emo_true_y_list, 
                cau_pred_y_list, cau_true_y_list, cau_pred_y_list_all, cau_true_y_list_all, cau_pred_y_list_all_windowed,
                emo_cause_pred_y_list, emo_cause_true_y_list, emo_cause_pred_y_list_all, emo_cause_true_y_list_all,
                pair_label_indegree_batch_window, pair_label_indegree_batch_all,
                loss_avg):
    
    if dataset_type == 'RECCON':
        label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
        label_neutral = 6
    elif dataset_type == 'ConvECPE':
        label_ = np.array(['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'])
        label_neutral = 2
        
    additional_report = ""
    emo_report_dict = metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_, get_dict=True)
    emo_report_str = metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_, get_dict=False)
    acc_emo, macro_f1, weighted_f1 = emo_report_dict['accuracy'], emo_report_dict['macro avg']['f1-score'], emo_report_dict['weighted avg']['f1-score']
    emo_metrics = (acc_emo, macro_f1, weighted_f1)
    
    emo_binary_str = metrics_report_for_emo_binary(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label_neutral=label_neutral, get_dict=False)
    emo_binary_dict = metrics_report_for_emo_binary(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label_neutral=label_neutral, get_dict=True)
    
    pred_y, true_y = argmax_prediction(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list))
    pred_y = [1 if element == label_neutral else 0 for element in pred_y]
    true_y = [1 if element == label_neutral else 0 for element in true_y]
    auc = roc_auc_score(true_y, pred_y)
    emo_binary_str += f'\n\n[Emotion Performance]\n'+\
                        f'\n<Multiclass>'+\
                        f'\n\tacc_emo:     \t{acc_emo}'+\
                        f'\n\tmacro_f1:    \t{macro_f1}'+\
                        f'\n\tweighted_f1: \t{weighted_f1}'+\
                        f'\n\n<Binary>'+\
                        f"\n\tAUC:         \t{auc}"+\
                        f'\n'
                        
    label_ = np.array(['No Cause', 'Cause'])
    report_dict = metrics_report(torch.cat(cau_pred_y_list), torch.cat(cau_true_y_list), label=label_, get_dict=True)
    if 'Cause' in report_dict.keys():  
        _, p_cau, _, _ = report_dict['accuracy'], report_dict['Cause']['precision'], report_dict['Cause']['recall'], report_dict['Cause']['f1-score']
    else:
        _, p_cau, _, _ = 0, 0, 0, 0
        
    
    class_name = list(label_)
    report_dict = classification_report(torch.cat(cau_true_y_list_all).cpu(), torch.cat(cau_pred_y_list_all_windowed).cpu(), target_names=class_name, zero_division=0, digits=4, output_dict=True)
    
    if 'Cause' in report_dict.keys():
        acc_cau, _, r_cau, _ = report_dict['accuracy'], report_dict['Cause']['precision'], report_dict['Cause']['recall'], report_dict['Cause']['f1-score']
    else:
        acc_cau, _, r_cau, _ = 0, 0, 0, 0
        
    f1_cau = 2 * p_cau * r_cau / (p_cau + r_cau) if p_cau + r_cau != 0 else 0
        
    if dataset_type == 'RECCON':
        label_ = np.array([1, 11, 21, 31, 41, 51, 61, 0, 10, 20, 30, 40, 50, 60])
        label_neutral = 6
    elif dataset_type == 'ConvECPE':
        label_ = np.array([1, 11, 21, 31, 41, 51, 0, 10, 20, 30, 40, 50])
        label_neutral = 2
        
    confusion_pred = confusion_matrix(torch.cat(emo_cause_true_y_list).to('cpu'), torch.cat(emo_cause_pred_y_list).to('cpu'), labels=label_)
    confusion_all = confusion_matrix(torch.cat(emo_cause_true_y_list_all).to('cpu'), torch.cat(emo_cause_pred_y_list_all).to('cpu'), labels=label_)
    
    idx_list = [0, 1, 2, 3, 4, 5]
    pred_num_acc_list = [0, 0, 0, 0, 0, 0]
    pred_num_acc_list_all = [0, 0, 0, 0, 0, 0]
    pred_num_precision_denominator_dict = [0, 0, 0, 0, 0, 0]
    pred_num_recall_denominator_dict = [0, 0, 0, 0, 0, 0]
    
    for i in idx_list:
        pred_num_acc_list[i] = confusion_pred[i][i]
        pred_num_acc_list_all[i] = confusion_all[i][i]
        pred_num_precision_denominator_dict[i] = sum(confusion_pred[:,i])
        pred_num_recall_denominator_dict[i] = sum(confusion_all[i,:])
    
    micro_precision = sum(pred_num_acc_list) / sum(pred_num_precision_denominator_dict)
    micro_recall = sum(pred_num_acc_list) / sum(pred_num_recall_denominator_dict)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall != 0 else 0
    
    p_emo_cau = micro_precision
    r_emo_cau = micro_recall
    f1_emo_cau = micro_f1
        
    return emo_report_str, emo_metrics, emo_binary_str, additional_report, acc_cau, p_cau, r_cau, f1_cau, p_emo_cau, r_emo_cau, f1_emo_cau

def argmax_prediction(pred_y, true_y):
    pred_argmax = torch.argmax(pred_y, dim=1).cpu()
    true_y = true_y.cpu()
    return pred_argmax, true_y

def threshold_prediction(pred_y, true_y):
    pred_y = pred_y > 0.5
    return pred_y, true_y

def metrics_report(pred_y, true_y, label, get_dict=False, multilabel=False):
    true_y = true_y.view(-1)
    if multilabel:
        pred_y, true_y = threshold_prediction(pred_y, true_y)
        available_label = sorted(list(set((pred_y == True).nonzero()[:, -1].tolist() + (true_y == True).nonzero()[:, -1].tolist())))
    else:
        pred_y, true_y = argmax_prediction(pred_y, true_y)
        available_label = sorted(list(set(true_y.tolist() + pred_y.tolist())))

    class_name = list(label[available_label])
    if get_dict:
        return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4, output_dict=True)
    else:
        return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4)
