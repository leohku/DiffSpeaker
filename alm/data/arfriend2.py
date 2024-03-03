from .base import BASEDataModule
from alm.data.ARFriend2 import ARFriend2Dataset
from transformers import Wav2Vec2Processor
from collections import defaultdict

import os
import re
from os.path import join as pjoin
import copy
import pickle
from tqdm import tqdm
import librosa
import numpy as np
from multiprocessing import Pool
import math

def load_data(args):
    files, root_dir, processor, templates, audio_dir, vertice_dir = args
    name = files[0].split('_')[0] + '_' + files[0].split('.')[0].split('_')[2]
    # Leo: temp hack to deal with /dev/shm memory limit
    # if files[0].startswith('20240128'):
    #     print("Skipping " + name)
    #     return None
    print("Loading data for " + name)
    wav_paths = [os.path.join(root_dir, audio_dir, f.replace(".npy", ".wav")) for f in files]
    speech_arrays = [librosa.load(wp, sr=16000)[0] for wp in wav_paths]
    input_values = [np.squeeze(processor(sa, sampling_rate=16000).input_values) for sa in speech_arrays]
    if input_values[0].shape[0] != input_values[1].shape[0]:
        length = min(input_values[0].shape[0], input_values[1].shape[0])
        input_values = [array[:length] for array in input_values]
    templates = [templates[f.split('_')[1]].reshape((-1)) for f in files]
    male_index = 0 if files[0].split('_')[1] in ['001Sky', '003Alan', '005Richard', '008Kunio'] else 1
    result = {
        "name": name,
        "path": files[0],
        "audio": input_values,
        "template": templates,
        "vertice_path": files,
        "male_index": male_index
    }
    # print("Data loaded for " + name)
    return (name, result)

class ARFriend2DataModule(BASEDataModule):
    def __init__(self,
                cfg,
                batch_size,
                num_workers,
                collate_fn = None,
                phase="train",
                multimodal=False,
                **kwargs):
        super().__init__(batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=collate_fn)
        self.save_hyperparameters(logger=False)
        self.name = 'ARFriend2'
        self.Dataset = ARFriend2Dataset
        self.cfg = cfg
        
        self.dates = {
            'train': [
                '20231119',
                '20231126',
                '20231208',
                '20240126',
                '20240128'
            ],
            'val': [
                '20240126'
            ],
            'test': [
                '20240126'
            ]
        }

        self.root_dir = kwargs.get('data_root', 'datasets/arfriend')
        self.audio_dir = 'wav'
        self.vertice_dir = 'vertices_npy'
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.template_file = 'templates.pkl'
        self.rel_file = 'rel.pkl'

        self.nfeats = 72147*2
        self.segmented_append_seconds = 5

        # load
        data = defaultdict(dict)
        with open(os.path.join(self.root_dir, self.template_file), 'rb') as fin:
            templates = pickle.load(fin, encoding='latin1')
        
        if multimodal:
            with open(os.path.join(self.root_dir, self.rel_file), 'rb') as fin:
                rel = pickle.load(fin)
        
        args_list = []
        processed_pairs = []
        vertice_files = os.listdir(os.path.join(self.root_dir, self.vertice_dir))
        for file in vertice_files:
            result = re.search('^(.+)_(.+)_(.+).npy$', file)
            if (result.group(1), result.group(3)) in processed_pairs:
                continue
            processed_pairs.append((result.group(1), result.group(3)))
            alt_file = next(iter([f for f in vertice_files if re.search(f'^{result.group(1)}_(.+)_{result.group(3)}.npy$', f) and f.split('_')[1] != result.group(2)]), None)
            if alt_file is None:
                continue
            args_list.append(([file, alt_file], self.root_dir, processor, templates, self.audio_dir, self.vertice_dir))

        with Pool(processes=os.cpu_count()) as pool:
            results = pool.map(load_data, args_list)
            for result in results:
                if result is not None:
                    key, value = result
                    data[key] = value

        splits = {
                    'train':range(33,970),
                    'val':range(1,33),
                    'test':range(1,33)
                }

        # split dataset
        self.data_splits = {
            'train':[],
            'val':[],
            'test':[],
        }
        
        def segmented_append(data_list, orig_v, seconds=5):
            assert orig_v["audio"][0].shape[0] == orig_v["audio"][1].shape[0]
            audio_ticks = orig_v["audio"][0].shape[0]
            for i in range(math.ceil(audio_ticks / (16000 * seconds))):
                vertice_paths = [f"/dev/shm/vertices_npy_{seconds}seg/{v.split('.')[0]}-{str(i)}.npy" for v in orig_v["vertice_path"]]
                if (i+1) * 16000 * seconds <= audio_ticks:
                    audio = [a[i * 16000 * seconds : (i+1) * 16000 * seconds] for a in orig_v["audio"]]
                else:
                    audio = [a[i * 16000 * seconds :] for a in orig_v["audio"]]
                new_v = copy.copy(orig_v)
                new_v["segment"] = i
                new_v["audio"] = audio
                new_v["vertice_path"] = vertice_paths
                if multimodal:
                    if len(rel[new_v["name"]]) <= i: # out-of-bounds protection
                        new_v["rel"] = rel[new_v["name"]][len(rel[new_v["name"]]) - 1]
                    else:
                        new_v["rel"] = rel[new_v["name"]][i]
                data_list.append(new_v)
    
        for k, v in data.items():
            date = k.split("_")[0]
            sentence_id = int(k.split("_")[1])
            for env in ['train', 'val', 'test']:
                if date in self.dates[env] and sentence_id in splits[env]:
                    segmented_append(self.data_splits[env], v, seconds=self.segmented_append_seconds)

        print("Data splits stats:")
        print(len(self.data_splits['train']))
        print(len(self.data_splits['val']))
        print(len(self.data_splits['test']))
        print(f"Multimodal: {'on' if multimodal else 'off'}")

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
                    dates_dict = self.dates,
                    data_type = subset,
                    segmented_append_seconds = self.segmented_append_seconds
                )
            return getattr(self, item_c)
        classname = self.__class__.__name__
        raise AttributeError(f"'{classname}' object has no attribute '{item}'")