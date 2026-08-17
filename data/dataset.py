import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):

    """Holds pre-tokenized source/target id sequences."""

    def __init__(self, src_ids_list, tgt_ids_list):
        assert len(src_ids_list) == len(tgt_ids_list)
        self.src_ids_list = src_ids_list
        self.tgt_ids_list = tgt_ids_list

    def __len__(self):
        return len(self.src_ids_list)

    def __getitem__(self, idx):
        return self.src_ids_list[idx], self.tgt_ids_list[idx]


def make_collate_fn(pad_id: int, bos_id: int, eos_id: int, max_len: int = 128):
    
    """Builds padded batches with BOS-shifted decoder inputs."""

    def collate(batch):
        src_batch = [ids[:max_len] for ids, _ in batch]
        tgt_batch = [ids[: max_len - 1] for _, ids in batch] 

        src_len = max(len(ids) for ids in src_batch)
        dec_len = max(len(ids) for ids in tgt_batch) + 1

        batch_size = len(batch)
        src_ids = torch.full((batch_size, src_len), pad_id, dtype=torch.long)
        src_mask = torch.zeros((batch_size, src_len), dtype=torch.long)
        decoder_input = torch.full((batch_size, dec_len), pad_id, dtype=torch.long)
        labels = torch.full((batch_size, dec_len), pad_id, dtype=torch.long)
        tgt_mask = torch.zeros((batch_size, dec_len), dtype=torch.long)

        for i, ids in enumerate(src_batch):
            src_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            src_mask[i, : len(ids)] = 1

        for i, ids in enumerate(tgt_batch):
            dec_in = [bos_id] + ids
            label = ids + [eos_id]
            decoder_input[i, : len(dec_in)] = torch.tensor(dec_in, dtype=torch.long)
            labels[i, : len(label)] = torch.tensor(label, dtype=torch.long)
            tgt_mask[i, : len(dec_in)] = 1

        return {
            "src_ids": src_ids,
            "src_mask": src_mask,
            "decoder_input": decoder_input,
            "tgt_mask": tgt_mask,
            "labels": labels,
        }

    return collate
