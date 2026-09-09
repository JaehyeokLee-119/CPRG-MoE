# ConvECPE dataset preparation

We do not redistribute the original dataset because the
[IEMOCAP license](https://sail.usc.edu/iemocap/Data_Release_Form_IEMOCAP.pdf)
does not permit sharing the data without permission from the University of
Southern California. Follow the steps below to obtain the source file and
construct the processed ConvECPE dataset locally.

Run the following commands from the repository's `data` directory.

## 1. Download the original dataset

```bash
curl -L \
  https://raw.githubusercontent.com/SenticNet/ECPEC/main/Dataset/IEMOCAP_emotion_cause_features.pkl \
  -o data_ConvECPE/IEMOCAP_emotion_cause_features.pkl
```

The `raw.githubusercontent.com` URL downloads the file directly. The source can
also be viewed on its
[GitHub repository page](https://github.com/SenticNet/ECPEC/blob/main/Dataset/IEMOCAP_emotion_cause_features.pkl).

## 2. Process the dataset

```bash
python data_ConvECPE/process_ConvECPE.py \
  --source data_ConvECPE/IEMOCAP_emotion_cause_features.pkl \
  --output-dir data_ConvECPE
```

This creates five fold directories, `data_ConvECPE/data_0` through
`data_ConvECPE/data_4`. Each directory contains training, validation, and test
JSON files.

> **Note:** If `data_ConvECPE` already contains generated files with the same
> names, this command overwrites them.
