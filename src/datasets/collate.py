import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    dataset_items = sorted(dataset_items, key=lambda x: x["spectrogram"].shape[2], reverse=True)

    result_batch['audio'] = pad_sequence(
        [sample["audio"].squeeze(0) for sample in dataset_items], batch_first=True
    )

    result_batch['spectrogram_length'] = torch.tensor([sample['spectrogram'].shape[2] for sample in dataset_items])

    # (batch_size, n_mels, time)
    result_batch['spectrogram'] = pad_sequence(
        [sample['spectrogram'].squeeze(0).permute(1, 0) for sample in dataset_items],
        batch_first=True
    ).permute(0, 2, 1)


    texts = [sample['text_encoded'].squeeze(0) for sample in dataset_items]
    # (num_samples, max_len)
    result_batch['text_encoded'] = pad_sequence(texts, batch_first=True)

    result_batch['text_encoded_length'] = torch.tensor([len(text) for text in texts])

    result_batch["text"] = [sample["text"] for sample in dataset_items]
    result_batch["audio_path"] = [sample["audio_path"] for sample in dataset_items]

    return result_batch