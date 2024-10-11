# Automatic speech recognition with DeepSpeech2

Implementation is based on the [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/pdf/1512.02595.pdf) article.

The model recognizes speech in audio.

You can read the original [statement](https://github.com/markovka17/dla/tree/2024/hw1_asr).

## Installation guide

1. Install libraries
```shell
pip install -r requirements.txt
```
2. Insall LM helpers
```shell
sh ./download_lm_prerequisites.sh
```
3. Install best model checkpoint
```shell
sh ./download_best_model.sh
```

## Train
```shell
python train.py
```

## Inference
```shell
python inference.py
```

## Wandb report
You can read my wandb [report]() (Russian only).