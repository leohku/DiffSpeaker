import numpy as np
import torch
from torch.utils import data
from transformers import Wav2Vec2Processor
from collections import defaultdict
import os
from tqdm import tqdm
import numpy as np
import pickle



class ARFriend2Dataset(data.Dataset):

    def __init__(self, 
                data, 
                dates_dict,
                segmented_append_seconds,
                data_type="train",
                ):

        self.data = data
        self.len = len(self.data)
        self.dates_dict = dates_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(dates_dict["train"]))
        self.segmented_append_seconds = segmented_append_seconds

    def __getitem__(self, index):
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        file_path = self.data[index]["path"]
        audio_male = self.data[index]["audio"][self.data[index]["male_index"]]
        audio_female = self.data[index]["audio"][1 - self.data[index]["male_index"]]
        vertice_male = np.load(self.data[index]["vertice_path"][self.data[index]["male_index"]])
        vertice_female = np.load(self.data[index]["vertice_path"][1 - self.data[index]["male_index"]])
        if vertice_male.shape[0] != vertice_female.shape[0]:
            length = min(vertice_male.shape[0], vertice_female.shape[0])
            vertice_male = vertice_male[:length]
            vertice_female = vertice_female[:length]
        vertice = np.hstack((vertice_male, vertice_female))
        template = np.concatenate((self.data[index]["template"][self.data[index]["male_index"]], self.data[index]["template"][1 - self.data[index]["male_index"]]))

        if "rel" in self.data[index]:
            rel = np.array(self.data[index]["rel"])
        
        if self.data_type == "train":
            date = file_name.split("_")[0]
            one_hot = self.one_hot_labels[self.dates_dict["train"].index(date)]
        elif self.data_type == "val":
            one_hot = self.one_hot_labels
        elif self.data_type == "test":
            date = file_name.split("_")[0]
            if date in self.dates_dict["train"]:
                one_hot = self.one_hot_labels[self.dates_dict["train"].index(date)]
            else:
                one_hot = self.one_hot_labels

        item = {
            'audio_male':torch.FloatTensor(audio_male),
            'audio_female':torch.FloatTensor(audio_female),
            'audio_attention':torch.ones_like(torch.Tensor(audio_male)).long(),
            'vertice':torch.FloatTensor(vertice), 
            'vertice_attention':torch.ones_like(torch.Tensor(vertice)[..., 0]).long(),
            'template':torch.FloatTensor(template), 
            'id':torch.FloatTensor(one_hot), 
            'file_name':file_name,
            'file_path':file_path
        }
        
        if "rel" in self.data[index]:
            item['rel'] = torch.FloatTensor(rel)
        
        return item

    def __len__(self):
        return self.len
    