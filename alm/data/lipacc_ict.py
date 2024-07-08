from .base import BASEDataModule
from alm.data.LipAccICT import LipAccICTDataset
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
            # result["vertice_path"] = vertice_path
            # result["vertice"] = np.load(vertice_path,allow_pickle=True)
            print("Data loaded for " + file)
            return (key, result)

class LipAccICTDataModule(BASEDataModule):
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
        self.name = 'LipAcc-ICT'
        self.Dataset = LipAccICTDataset
        self.cfg = cfg
        
        # customized to LipAcc-ICT
        self.subjects = {
            'train': [
                '006Vasilisa'
            ],
            'val': [
                '006Vasilisa'
            ],
            'test': [
                '006Vasilisa'
            ]
        }

        self.root_dir = kwargs.get('data_root', 'datasets/lipacc')
        self.audio_dir = 'wav'
        self.vertice_dir = 'vertices_npy_ict'
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.template_file = 'templates_ict.pkl'

        self.nfeats = 42186
        self.segmented_append_seconds = 5

        # load
        data = defaultdict(dict)
        with open(os.path.join(self.root_dir, self.template_file), 'rb') as fin:
            templates = pickle.load(fin, encoding='latin1')

        count = 0
        args_list = []
        for r, ds, fs in os.walk(os.path.join(self.root_dir, self.audio_dir)):
            for f in fs:
                args_list.append((f, self.root_dir, processor, templates, self.audio_dir, self.vertice_dir, ))

                # # comment off for full dataset
                # count += 1
                # if count > 10:
                #     break


        if True: # multi-process
            with Pool(processes=os.cpu_count()) as pool:
                results = pool.map(load_data, args_list)
                for result in results:
                    if result is not None:
                        key, value = result
                        data[key] = value
        else: # single process
            for args in tqdm(args_list, desc="Loading data"):
                result = load_data(args)
                if result is not None:
                    key, value = result
                    data[key] = value
                else:
                    print("Warning: data not found")


        # # calculate mean and std
        # motion_list = np.concatenate(motion_list, axis=0)
        # self.mean = np.mean(motion_list, axis=0)
        # self.std = np.std(motion_list, axis=0)

        splits = {
                    'train':range(2,44),
                    'val':range(44,47),
                    'test':range(44,47)
                }

        # split dataset
        self.data_splits = {
            'train':[],
            'val':[],
            'test':[],
        }
        
        # for k, v in data.items():
        #     subject_id = k.split("_")[1]
        #     sentence_id = int(k.split(".")[0][-3:])
        #     for sub in ['train', 'val', 'test']:
        #         if subject_id in self.subjects[sub] and sentence_id in splits[sub]:
        #             self.data_splits[sub].append(v)
        
        def segmented_append(data_list, orig_v, seconds=5):
            audio_ticks = orig_v["audio"].shape[0]
            for i in range(math.ceil(audio_ticks / (16000 * seconds))):
                name = orig_v["name"]
                new_v = defaultdict(dict)
                new_v["segment"] = i
                new_v["name"] = name
                new_v["vertice_path"] = f"/dev/shm/vertices_npy_ict_{seconds}seg/{name}-{str(i)}.npy"
                new_v["path"] = orig_v["path"]
                new_v["template"] = orig_v["template"]
                if (i+1) * 16000 * seconds <= audio_ticks:
                    new_v["audio"] = orig_v["audio"][i * 16000 * seconds : (i+1) * 16000 * seconds]
                    # new_v["vertice"] = orig_v["vertice"][i * 30 * seconds : (i+1) * 30 * seconds]
                else:
                    new_v["audio"] = orig_v["audio"][i * 16000 * seconds :]
                    # new_v["vertice"] = orig_v["vertice"][i * 30 * seconds :]
                data_list.append(new_v)
    
        for k, v in data.items():
            subject_id = k.split("_")[1]
            sentence_id = int(k.split(".")[0][-3:])
            for sub in ['train', 'val', 'test']:
                if subject_id in self.subjects[sub] and sentence_id in splits[sub]:
                    segmented_append(self.data_splits[sub], v, seconds=self.segmented_append_seconds)

        # self._sample_set = self.__getattr__("test_dataset")
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