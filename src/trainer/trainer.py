from pathlib import Path
import wandb

import pandas as pd
import numpy as np

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, raw_spectrogram, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

        spectrogram_for_plot = raw_spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("raw_spectrogram", image)

    def log_predictions(
        self, normalized_text, log_probs, log_probs_length, audio_path, audio, raw_audio, examples_to_log=10, **batch
    ):
        lengths = log_probs_length.detach().numpy()
        rows = {}
        for i in range(min(examples_to_log, len(normalized_text))):
            log_prob = log_probs[i].detach().cpu()
            log_prob_length = log_probs_length[i].detach().cpu().numpy()

            decode_method_to_func = {
                'argmax': self.text_encoder.argmax_ctc_decode,
                'bs': self.text_encoder.ctc_beam_search,
                'bs_lm': self.text_encoder.lib_lm_beam_search
            }

            input = {
                'probs': log_prob[:log_prob_length, :],
                'probs_lengths': np.array([log_prob_length])
            }

            target_text = normalized_text[i]

            rows[Path(audio_path[i]).name] = {
                "target": target_text,
                "raw_audio": wandb.Audio(raw_audio[i], sample_rate=16000),
                "audio": wandb.Audio(audio[i], sample_rate=16000)
            }

            for decode_method in self.saver_decode_methods:
                assert decode_method in list(decode_method_to_func.keys()), 'unknown decode method'

                predicted_text = decode_method_to_func[decode_method](**input)
                if not isinstance(predicted_text, str):
                    predicted_text = predicted_text[0]
                # saving_path.write_text(predicted_text, encoding='utf-8')

                wer = calc_wer(target_text, predicted_text) * 100
                cer = calc_cer(target_text, predicted_text) * 100


                rows[Path(audio_path[i]).name][f'{decode_method}_predictions'] = predicted_text
                rows[Path(audio_path[i]).name][f'{decode_method}_wer'] = wer
                rows[Path(audio_path[i]).name][f'{decode_method}_cer'] = cer

        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
