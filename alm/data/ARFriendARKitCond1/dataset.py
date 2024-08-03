import numpy as np
import torch
from torch.utils import data
from transformers import Wav2Vec2Processor
from collections import defaultdict
import os
from tqdm import tqdm
import numpy as np
import pickle



class ARFriendARKitCond1Dataset(data.Dataset):

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
        audio_cond = self.data[index]["audio_cond"]
        vertice = np.load(self.data[index]["vertice_path"])
        vertice_cond = np.load(self.data[index]["vertice_cond_path"])
        if self.data_type == "train":
            subject = file_name.split("_")[1]
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        elif self.data_type == "val":
            one_hot = self.one_hot_labels
        elif self.data_type == "test":
            subject = file_name.split("_")[1]
            if subject in self.subjects_dict["train"]:
                one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
            else:
                one_hot = self.one_hot_labels
        
        # truncate or extend audio_cond if size of audio and audio_cond is different
        if audio.shape != audio_cond.shape:
            if audio.shape[0] > audio_cond.shape[0]:
                audio_cond = np.pad(audio_cond, (0, audio.shape[0] - audio_cond.shape[0]), 'constant', constant_values=0)
            else:
                audio_cond = audio_cond[:audio.shape[0]]
        
        # truncate or extend vertice_cond if size of vertice and vertice_cond is different
        if vertice.shape != vertice_cond.shape:
            if vertice.shape[0] > vertice_cond.shape[0]:
                last_frame = vertice_cond[-1]
                padding = np.tile(last_frame, (vertice.shape[0] - vertice_cond.shape[0], 1))
                vertice_cond = np.concatenate((vertice_cond, padding), axis=0)
            else:
                vertice_cond = vertice_cond[:vertice.shape[0]]

        result = {
            'audio':torch.FloatTensor(audio),
            'audio_cond':torch.FloatTensor(audio_cond),
            'audio_attention':torch.ones_like(torch.Tensor(audio)).long(),
            'vertice':torch.FloatTensor(vertice), 
            'vertice_cond':torch.FloatTensor(vertice_cond),
            'vertice_attention':torch.ones_like(torch.Tensor(vertice)[..., 0]).long(),
            'id':torch.FloatTensor(one_hot), 
            'file_name':file_name,
            'file_path':file_path
        }
        
        return result

    def __len__(self):
        return self.len
    