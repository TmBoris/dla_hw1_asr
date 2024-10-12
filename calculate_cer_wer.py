from pathlib import Path
from src.metrics.utils import calc_wer, calc_cer
import hydra

@hydra.main(version_base=None, config_path="src/configs", config_name="cer_wer")
def main(config):
    target_path = config.target_path
    predictions_path = config.predictions_path

    wers = []
    cers = []
    dir_length = len(list(Path(target_path).iterdir()))
    for i in range(dir_length):
        with open(Path(target_path) / f'Utterance_{i}.txt', 'r') as f:
            target_text = f.read().strip()

        with open(Path(predictions_path) / f'Utterance_{i}.txt', 'r') as f:
            predicted_text = f.read().strip()

        wer = calc_wer(target_text, predicted_text) * 100
        cer = calc_cer(target_text, predicted_text) * 100

        wers.append(wer)
        cers.append(cer)

    print('Total WER:', sum(wers) / len(wers))
    print('Total CER:', sum(cers) / len(cers))


if __name__ == "__main__":
    main()