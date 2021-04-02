import torch
from torch.utils.data import Dataset


class BertDataset(Dataset):
    def __init__(self, movie_plots, labels, tokenizer, blocksize=50):
        self.input_ids = []
        self.attention_masks = []
        self.labels = labels

        for plot in movie_plots:
            encode_dict = tokenizer.encode_plus(
                plot,
                max_length=blocksize,
                padding='max_length',
                truncation=True,
                return_attention_mask=True
            )
            self.input_ids.append(torch.tensor(encode_dict['input_ids'], dtype=torch.long).flatten())
            self.attention_masks.append(torch.tensor(encode_dict['attention_mask'], dtype=torch.long).flatten())
            self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return self.input_ids[i], self.attention_masks[i], self.labels[i]
