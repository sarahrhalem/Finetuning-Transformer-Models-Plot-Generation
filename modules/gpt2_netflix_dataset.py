import os
import torch
from torch.utils.data import Dataset


# Dataset Class
class NetflixPlotDataset(Dataset):
    def __init__(self, tokenizer, dataset_path=os.path.join("Data\\netflix_plot_dataset.txt"),
                 block_size=50):  # Block size set to max length of data

        bos_tkn = tokenizer.bos_token
        sep_tkn = tokenizer.sep_token
        eos_tkn = tokenizer.eos_token

        # Load dataset txt file line by line
        plot_file = open(os.path.join(dataset_path), encoding="utf-8")
        lines = []
        for line in plot_file.read().splitlines():
            if len(line) > 0 and not line.isspace():
                try:
                    lines += [bos_tkn + line[:line.index(':')] + " :" + sep_tkn + line[line.index(':') + 1:] + eos_tkn]
                except ValueError:
                    raise ValueError('line ' + line + ' does not contain ":"')

        # Encode plot data with addition of special tokens for line by line loading
        self.input_ids = tokenizer.batch_encode_plus(lines, add_special_tokens=True,
                                                     max_length=block_size, truncation=True,
                                                     padding='max_length')["input_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return torch.tensor(self.input_ids[i], dtype=torch.long)
