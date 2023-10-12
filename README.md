# CPRG-MoE

Code for CPRG-MoE (Enhancing Emotion–Cause Pair Extraction in Conversation With Contextual Information) <br>
[PRG-MoE](https://github.com/jdjin3000/PRG-MoE) referenced

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

## Dataset
The dataset used in this model is [RECCON dataset](https://github.com/declare-lab/RECCON) and [ConvECPE dataset](https://github.com/Maxwe11y/JointEC/tree/main/Dataset)

```bash
python main.py --gpus 0,1 --num_process 8 --num_worker 6 --model_name CPRG_MoE --train_data data/data_ConvECPE/data_0/ConvECPE_fold_0_train.json --valid_data data/data_ConvECPE/data_0/ConvECPE_fold_0_valid.json --test_data data/data_ConvECPE/data_0/ConvECPE_fold_0_test.json --dataset_type ConvECPE --data_label ConvECPE_fold_0 --n_emotion 6 --log_directory logs --ckpt_filename ckpt --encoder_name bert-base-cased --max_seq_len 128 --loss_lambda 0.2 --dropout 0.5 --guiding_lambda 0.6 --training_iter 15 --batch_size 4 --learning_rate 5e-5
```
