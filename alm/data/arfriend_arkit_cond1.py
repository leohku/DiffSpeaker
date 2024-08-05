from .base import BASEDataModule
from alm.data.ARFriendARKitCond1 import ARFriendARKitCond1Dataset
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

PAIRS = {
    '20231119': ['001Sky', '002Shirley'],
    '20231126': ['003Alan', '007Jessica'],
    '20231208': ['005Richard', '006Vasilisa'],
    '20240126': ['008Kunio', '006Vasilisa'],
    '20240128': ['001Sky', '007Jessica']
}

def load_data(args):
    file, root_dir, processor, audio_dir, vertice_dir = args
    print("Loading data for " + file)
    if file.endswith('wav'):
        wav_path = os.path.join(root_dir, audio_dir, file)
        speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
        input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
        
        # find out conditioning information
        date = file.split("_")[0]
        cond_actor_idx = PAIRS[date].index(file.split("_")[1]) - 1
        cond_actor = PAIRS[date][cond_actor_idx]
        scenario_id = file.split("_")[2].split(".")[0]
        wav_cond_path = os.path.join(root_dir, audio_dir, f"{date}_{cond_actor}_{scenario_id}.wav")
        if not os.path.exists(wav_cond_path):
            print("No conditioning audio found for " + file)
            return None
        cond_input_values, sampling_rate = librosa.load(wav_cond_path, sr=16000)
        cond_input_values = np.squeeze(processor(cond_input_values,sampling_rate=16000).input_values)
        
        key = file.replace(".wav", "")
        result = {}
        result["audio"] = input_values
        result["audio_cond"] = cond_input_values
        result["name"] = key
        result["cond_name"] = f"{date}_{cond_actor}_{scenario_id}"
        result["path"] = os.path.abspath(wav_path)
        print("Main and conditioning audio loaded for " + file)
        return (key, result)

class ARFriendARKitCond1DataModule(BASEDataModule):
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
        self.name = 'ARFriendARKitCond1'
        self.Dataset = ARFriendARKitCond1Dataset
        self.cfg = cfg
        
        self.subjects = {
            'train': [
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
        self.vertice_dir = 'arkit_npy'
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.train_list_file = 'train_list.txt'

        self.nfeats = 55
        self.segmented_append_seconds = 5
        
        with open(os.path.join(self.root_dir, self.train_list_file), 'r') as fin:
            train_list = [line.strip() for line in fin]

        # load
        data = defaultdict(dict)

        count = 0
        args_list = []
        for r, ds, fs in os.walk(os.path.join(self.root_dir, self.audio_dir)):
            for f in fs:
                date = f.split("_")[0]
                scenario_id = f.split(".")[0].split("_")[-1].zfill(3)
                if f'{date}_{scenario_id}' in train_list:
                    args_list.append((f, self.root_dir, processor, self.audio_dir, self.vertice_dir, ))

                # comment off for full dataset
                # count += 1
                # if count > 10:
                #     break


        with Pool(processes=os.cpu_count()) as pool:
            results = pool.map(load_data, args_list)
            for result in results:
                if result is not None:
                    key, value = result
                    data[key] = value


        # # calculate mean and std
        # motion_list = np.concatenate(motion_list, axis=0)
        # self.mean = np.mean(motion_list, axis=0)
        # self.std = np.std(motion_list, axis=0)

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
                new_v["cond_name"] = orig_v["cond_name"]
                new_v["vertice_path"] = f"/dev/shm/arkit_npy_{seconds}seg/{name}-{str(i)}.npy"
                new_v["vertice_cond_path"] = f"/dev/shm/arkit_npy_{seconds}seg/{orig_v['cond_name']}-{str(i)}.npy"
                # check if both files exist
                if not os.path.exists(new_v["vertice_path"]):
                    print("Warning: vertice file for " + name + " not found")
                    continue
                if not os.path.exists(new_v["vertice_cond_path"]):
                    print("Warning: vertice file for " + orig_v["cond_name"] + ", as a condition to " + name + ", not found")
                    continue
                new_v["path"] = orig_v["path"]
                if (i+1) * 16000 * seconds <= audio_ticks:
                    new_v["audio"] = orig_v["audio"][i * 16000 * seconds : (i+1) * 16000 * seconds]
                    new_v["audio_cond"] = orig_v["audio_cond"][i * 16000 * seconds : (i+1) * 16000 * seconds]
                else:
                    new_v["audio"] = orig_v["audio"][i * 16000 * seconds :]
                    new_v["audio_cond"] = orig_v["audio_cond"][i * 16000 * seconds :]
                    # skip if audio is too short, to prevent
                    # ValueError: `mask_length` has to be smaller than `sequence_length`, but got `mask_length`: 10 and `sequence_length`: x`
                    if new_v["audio"].shape[0] / 16000 < 1 or new_v["audio_cond"].shape[0] / 16000 < 1:
                        continue
                data_list.append(new_v)
    
        for k, v in data.items():
            for sub in ['train']:
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