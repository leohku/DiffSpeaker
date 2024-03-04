from .base import BASEDataModule
from alm.data.ARFriend import ARFriendDataset
from transformers import Wav2Vec2Processor
from collections import defaultdict

import os
from os.path import join as pjoin
import pickle
from tqdm import tqdm
import librosa
import numpy as np
from multiprocessing import Pool
import math


def load_data(args):
    file, root_dir, processor, templates, audio_dir, vertice_dir = args
    # Leo: temp hack to deal with /dev/shm memory limit
    # if file.startswith('20240128'):
    #     print("Skipping " + file)
    #     return None
    print("Loading data for " + file)
    if file.endswith('wav'):
        wav_path = os.path.join(root_dir, audio_dir, file)
        speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
        input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
        key = file.replace("wav", "npy")
        result = {}
        result["audio"] = input_values
        subject_id = key.split("_")[1]
        temp = templates[subject_id]
        result["name"] = file.replace(".wav", "")
        result["path"] = os.path.abspath(wav_path)
        result["template"] = temp.reshape((-1)) 
        vertice_path = os.path.join(root_dir, vertice_dir, file.replace("wav", "npy"))
        if not os.path.exists(vertice_path):
            print("No vertices exist for " + file)
            return None
        else:
            print("Data loaded for " + file)
            return (key, result)

class ARFriendDataModule(BASEDataModule):
    def __init__(self,
                cfg,
                batch_size,
                num_workers,
                collate_fn = None,
                phase="train",
                **kwargs):
        super().__init__(batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=collate_fn)
        self.save_hyperparameters(logger=False)
        self.name = 'ARFriend'
        self.Dataset = ARFriendDataset
        self.cfg = cfg
        
        # customized to ARFriend
        self.subjects = {
            'train': [
                '001Sky',
                '002Shirley',
                '003Alan',
                '005Richard',
                '006Vasilisa',
                '007Jessica',
                '008Kunio',
            ],
            'val': [
                '001Sky',
                '002Shirley',
                '003Alan',
                '005Richard',
                '006Vasilisa',
                '007Jessica',
                '008Kunio',
            ],
            'test': [
                '001Sky',
                '002Shirley',
                '003Alan',
                '005Richard',
                '006Vasilisa',
                '007Jessica',
                '008Kunio',
            ]
        }

        self.root_dir = kwargs.get('data_root', 'datasets/arfriend')
        self.audio_dir = 'wav'
        self.vertice_dir = 'vertices_npy'
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.template_file = 'templates.pkl'
        self.train_list_file = 'train_list.txt'
        self.test_list_file = 'test_list.txt'

        self.nfeats = 72147
        self.segmented_append_seconds = 5

        # load
        data = defaultdict(dict)
        with open(os.path.join(self.root_dir, self.template_file), 'rb') as fin:
            templates = pickle.load(fin, encoding='latin1')

        with open(os.path.join(self.root_dir, self.train_list_file), 'r') as fin:
            train_list = [line.strip() for line in fin]
        
        with open(os.path.join(self.root_dir, self.test_list_file), 'r') as fin:
            test_list = [line.strip() for line in fin]

        count = 0
        args_list = []
        for r, ds, fs in os.walk(os.path.join(self.root_dir, self.audio_dir)):
            for f in fs:
                args_list.append((f, self.root_dir, processor, templates, self.audio_dir, self.vertice_dir, ))

                # # comment off for full dataset
                # count += 1
                # if count > 10:
                #     break

        with Pool(processes=os.cpu_count()) as pool:
            results = pool.map(load_data, args_list)
            for result in results:
                if result is not None:
                    key, value = result
                    data[key] = value

        # numeric splits
        # splits = {
        #             'train':range(1,970),
        #             'val':range(306,638),
        #             'test':range(306,638)
        #         }

        # split dataset
        self.data_splits = {
            'train':[],
            'val':[],
            'test':[],
        }
        
        def segmented_append(data_list, orig_v, seconds=5):
            audio_ticks = orig_v["audio"].shape[0]
            for i in range(math.ceil(audio_ticks / (16000 * seconds))):
                name = orig_v["name"]
                new_v = defaultdict(dict)
                new_v["segment"] = i
                new_v["name"] = name
                new_v["vertice_path"] = f"/dev/shm/vertices_npy_{seconds}seg/{name}-{str(i)}.npy"
                new_v["path"] = orig_v["path"]
                new_v["template"] = orig_v["template"]
                if (i+1) * 16000 * seconds <= audio_ticks:
                    new_v["audio"] = orig_v["audio"][i * 16000 * seconds : (i+1) * 16000 * seconds]
                else:
                    new_v["audio"] = orig_v["audio"][i * 16000 * seconds :]
                data_list.append(new_v)

        # numeric splits
        # for k, v in data.items():
        #     subject_id = k.split("_")[1]
        #     sentence_id = int(k.split(".")[0][-3:])
        #     for sub in ['train', 'val', 'test']:
        #         if subject_id in self.subjects[sub] and sentence_id in splits[sub]:
        #             segmented_append(self.data_splits[sub], v, seconds=self.segmented_append_seconds)
        
        for k, v in data.items():
            date = k.split("_")[0]
            subject_id = k.split("_")[1]
            sentence_id = k.split(".")[0][-3:]
            date_sentence = date + "_" + sentence_id
            
            if subject_id in self.subjects['train'] and date_sentence in train_list:
                segmented_append(self.data_splits['train'], v, seconds=self.segmented_append_seconds)
            if subject_id in self.subjects['val'] and date_sentence in test_list:
                segmented_append(self.data_splits['val'], v, seconds=self.segmented_append_seconds)
            if subject_id in self.subjects['test'] and date_sentence in test_list:
                segmented_append(self.data_splits['test'], v, seconds=self.segmented_append_seconds) 

        print("Data splits stats:")
        print(len(self.data_splits['train']))
        print(len(self.data_splits['val']))
        print(len(self.data_splits['test']))

    def __getattr__(self, item):
        # train_dataset/val_dataset etc cached like properties
        # question
        if item.endswith("_dataset") and not item.startswith("_"):
            subset = item[:-len("_dataset")]
            item_c = "_" + item
            if item_c not in self.__dict__:
                # todo: config name not consistent
                self.__dict__[item_c] = self.Dataset(
                    data = self.data_splits[subset] ,
                    subjects_dict = self.subjects,
                    data_type = subset,
                    segmented_append_seconds = self.segmented_append_seconds
                )
            return getattr(self, item_c)
        classname = self.__class__.__name__
        raise AttributeError(f"'{classname}' object has no attribute '{item}'")