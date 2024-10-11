import torch
import os
import wandb
import numpy as np
from tqdm.auto import tqdm
from scipy.io.wavfile import write
from pathlib import Path
import pandas as pd


from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        text_encoder,
        logger,
        writer,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            text_encoder (CTCTextEncoder): text encoder.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer
        self.max_logged_instances = self.cfg_trainer.get('max_logged_instances', 100)
        self.saver_decode_methods = self.cfg_trainer.get('saver_decode_methods', ['argmax'])

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        self.text_encoder = text_encoder

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition
        self.save_path = save_path
        
        self.writer = writer
        self.logger = logger
        self.log_to_wandb = True

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs
    
    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        # if self.save_path is not None:
        #     (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                if "test" not in part:
                    raise Exception(f"Evaluating part without 'test' prefics. part name is {part}")
                try:
                    batch = self.process_batch(
                        batch_idx=batch_idx,
                        batch=batch,
                        part=part,
                        metrics=self.evaluation_metrics,
                    )
                except torch.cuda.OutOfMemoryError as e:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()  # free some memory
                    continue

        return self.evaluation_metrics.result()

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """

        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(**batch)
        batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        # Some saving logic. This is an example
        # Use if you need to save predictions on disk

        prediction_pathes = {}
        for decode_method in self.saver_decode_methods:
            prediction_pathes[decode_method] = self.save_path
            os.makedirs(prediction_pathes[decode_method], exist_ok=True)

        for i in range(len(batch['audio'])):

            log_prob = batch['log_probs'][i].detach().cpu()
            log_prob_length = batch['log_probs_length'][i].detach().cpu().numpy()

            decode_method_to_func = {
                'argmax': self.text_encoder.argmax_ctc_decode,
                'bs': self.text_encoder.ctc_beam_search,
                'bs_lm': self.text_encoder.lib_lm_beam_search
            }

            input = {
                'probs': log_prob[:log_prob_length, :],
                'probs_lengths': np.array([log_prob_length])
            }

            cur_len = len(os.listdir(list(prediction_pathes.values())[0]))
            for decode_method in self.saver_decode_methods:
                assert decode_method in list(decode_method_to_func.keys()), 'unknown decode method'

                saving_path = prediction_pathes[decode_method] / f'Utterance_{cur_len}.txt'
                decoded_text = decode_method_to_func[decode_method](**input)
                if not isinstance(decoded_text, str):
                    decoded_text = decoded_text[0]
                saving_path.write_text(decoded_text, encoding='utf-8')
        if self.log_to_wandb:
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)
            self.log_to_wandb = False

        return batch
    
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
