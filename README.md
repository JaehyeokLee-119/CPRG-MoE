# CPRG-MoE
This code pertains to CPRG-MoE (Contextualized Pair-Relationship Guided Mixture-of-Experts), a model designed to enhance the extraction of emotion-cause pairs in conversations while considering contextual information. It is inspired by the [PRG-MoE repository](https://github.com/jdjin3000/PRG-MoE).

## Dataset
The model employs two datasets for training and evaluation: [RECCON dataset](https://github.com/declare-lab/RECCON) and the [ConvECPE dataset](https://github.com/Maxwe11y/JointEC/tree/main/Dataset)

## Dependencies
- python 3.9.17<br>
- pytorch 2.0.1<br>
- pytorch-cuda 11.8<br>
- numpy 1.23.5<br>
- huggingface_hub 0.16.4<br>
- cuda 11.8<br>
- transformers 4.29.2<br>
- scikit-learn 1.2.2<br>
- pytorch-lightning 2.0.5: pip install lightning<br> 

## How to run
1. Choose one of the data files from ConvECPE 0\~4 and RECCON 0\~4
2. Then, you can run the model by run below script with a few modifying on: [train_data, valid_data, test_data, dataset_type, data_label, n_emotion].<br><br>
* Set n_emotion to 7 if you are working with the RECCON dataset.
* Set n_emotion to 6 if you are working with the ConvECPE dataset.<br><br>
For example, if you choose ConvECPE fold 3:<br>
['/data/data_ConvECPE/data_3/ConvECPE_fold_3_train.json', '/data/data_ConvECPE/data_3/ConvECPE_fold_3_valid.json', '/data/data_ConvECPE/data_3/ConvECPE_fold_3_test.json', 'ConvECPE', 'ConvECPE_3', 6]<br><br>
If you opt for RECCON fold 0:<br>
['/data/data_RECCON/data_0/dailydialog_train.json', '/data/data_ConvECPE/data_0/dailydialog_valid.json', '/data/data_ConvECPE/data_0/dailydialog_test.json', 'RECCON', 'RECCON_0', 7]<br><br>
For RECCON fold 2:<br>
['/data/data_RECCON/data_2/data_2_train.json', '/data/data_RECCON/data_2/data_2_valid.json', '/data/data_RECCON/data_2/data_2_test.json', 'RECCON', 'RECCON_2', 7]<br><br>

```bash
python main.py \
    --gpus 1 \
    --num_process 8 \
    --num_worker 6 \
    --model_name CPRG_MoE \
    --train_data data/data_ConvECPE/data_0/ConvECPE_fold_0_train.json \
    --valid_data data/data_ConvECPE/data_0/ConvECPE_fold_0_valid.json \
    --test_data data/data_ConvECPE/data_0/ConvECPE_fold_0_test.json \
    --dataset_type ConvECPE \
    --data_label ConvECPE_fold_0 \
    --n_emotion 6 \
    --log_directory logs \
    --model_save_path ./model \
    --ckpt_filename ckpt \
    --max_seq_len 128 \
    --loss_lambda 0.2 \
    --dropout 0.5 \
    --guiding_lambda 0.6 \
    --training_iter 20 \
    --batch_size 4 \
    --learning_rate 5e-5
```


