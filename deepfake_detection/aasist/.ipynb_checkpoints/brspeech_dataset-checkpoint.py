from datasets import load_dataset, Audio
import torch
from torch.utils.data import Dataset
from data_utils import pad, pad_random

SR = 16000
NB_SAMP = 64600

class BRSpeechDataset(Dataset):

    def __init__(self, database_path, split, nb_samp):

        self.ds = load_dataset(str(database_path))
        self.split = list(self.ds.keys())[split]
        self.ds = self.ds[self.split].cast_column(
            "audio",
            Audio(sampling_rate=16000)
        )
        print(self.ds)
        bad_indices = set([206160, 260927, 352495])
        self.valid_indices = [i for i in range(len(self.ds))
            if i not in bad_indices]

        self.nb_samp = nb_samp

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):

        row = self.ds[self.valid_indices[idx]]

        wav = row["audio"]["array"]
        label = int(row["label"])

        # Match original AASIST preprocessing
        if self.split == 0: # 0 is train, 1 and 2 are validation and test respectively
            wav = pad_random(wav, self.nb_samp)
        else:
            wav = pad(wav, self.nb_samp)

        return torch.FloatTensor(wav), label