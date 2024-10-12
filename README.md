# Automatic speech recognition with DeepSpeech2

Implementation is based on the [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/pdf/1512.02595.pdf) article.

The model recognizes speech in audio.

You can read the original [statement](https://github.com/markovka17/dla/tree/2024/hw1_asr).

## Installation guide

0. (Optional) Create and activate new environment
   using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

   ```bash
   # create env
   conda create -n ASR python=3.10

   # activate env
   conda activate ASR
   ```

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

## Quick inference of the model on test examples and calculating metrics

    python inference.py
    python calculate_cer_wer.py
    

## Train model from scratch

    python train.py


## Wandb report
You can read my wandb [report](https://wandb.ai/tmboris/pytorch_template_asr_example/reports/DLA-HW1-ASR--Vmlldzo5NjkxNDMz?accessToken=j2a3oiwv4f4e69qjwc5lffy9b09alg4a2olr1d9du4b4p0p5c6sq51n383nmd2d2) (Russian only) for more details about the project and its developing.