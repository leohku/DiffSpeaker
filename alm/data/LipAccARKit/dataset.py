import numpy as np
import torch
from torch.utils import data
from transformers import Wav2Vec2Processor
from collections import defaultdict
import os
from tqdm import tqdm
import numpy as np
import pickle



class LipAccARKitDataset(data.Dataset):

    def __init__(self, 
                data, 
                subjects_dict,
                segmented_append_seconds,
                data_type="train",
                ):

        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        self.segmented_append_seconds = segmented_append_seconds

    def __getitem__(self, index):
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        file_path = self.data[index]["path"]
        audio = self.data[index]["audio"]
        vertice = np.load(self.data[index]["vertice_path"])
        if self.data_type == "train":
            subject = file_name.split("_")[1]
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
            # random sample between subjects
            # one_hot = self.one_hot_labels[np.random.randint(0, 7)]
        elif self.data_type == "val":
            one_hot = self.one_hot_labels
        elif self.data_type == "test":
            subject = file_name.split("_")[1]
            if subject in self.subjects_dict["train"]:
                one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
            else:
                one_hot = self.one_hot_labels

        return {
            'audio':torch.FloatTensor(audio),
            'audio_attention':torch.ones_like(torch.Tensor(audio)).long(),
            'vertice':torch.FloatTensor(vertice), 
            'vertice_attention':torch.ones_like(torch.Tensor(vertice)[..., 0]).long(),
            'id':torch.FloatTensor(one_hot), 
            'file_name':file_name,
            'file_path':file_path
        }

    def __len__(self):
        return self.len
    