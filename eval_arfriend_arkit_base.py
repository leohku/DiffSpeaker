import os
import pickle
import torch

from alm.config import parse_args
from alm.models.get_model import get_model
from alm.utils.logger import create_logger
from alm.utils.demo_utils import animate, load_example_input

import numpy as np

MODEL_CKPT = 'experiments/arfriend_arkit/diffusion_bias_arkit/diffspeaker_wav2vec2_arfriend_arkit_fine_base_cont/checkpoints/epoch=3429.ckpt'
EVAL_LIST = '/data3/leoho/arfriend-diffspeaker/test_list.txt'
WAV_PATH = '/data3/leoho/arfriend-diffspeaker/wav'
PRED_PATH = '/data6/leoho/arfriend-diffspeaker-ext/evals/base-test-set'

# paraterize the speaking style
speaker_to_id = {
    "001Sky": 0,
    "002Shirley": 1,
    "003Alan": 2,
    "005Richard": 3,
    "006Vasilisa": 4,
    "007Jessica": 5,
    "008Kunio": 6
}
date_subjects = {
    "20231119": ["001Sky", "002Shirley"],
    "20231126": ["003Alan", "007Jessica"],
    "20231208": ["005Richard"], # remove 006Vasilisa
    "20240126": ["008Kunio"], # remove 006Vasilisa
    "20240128": ["001Sky", "007Jessica"]
}

def main():
    cfg = parse_args(phase="demo")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    device = torch.device("cuda")
    
    dataset = 'arfriend_arkit'
    logger = create_logger(cfg, phase="demo")
    # set up the model architecture
    cfg.DATASET.NFEATS = 55
    model = get_model(cfg, dataset)
    
    # load model weights
    logger.info("Loading checkpoints from {}".format(MODEL_CKPT))
    state_dict = torch.load(MODEL_CKPT, map_location="cpu")["state_dict"]
    state_dict.pop("denoiser.PPE.pe") # this is not needed, since the sequence length can be any flexiable
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # load eval list
    eval_targets = []
    with open(EVAL_LIST, 'r') as f:
        for line in f:
            eval_targets.append(line.strip())
    
    for eval in eval_targets:
        
        date_id = eval.split('_')[0]
        scenario_id = eval.split('_')[1]
        
        for actor in date_subjects[date_id]:
            print(f'Predicting {date_id+"_"+actor+"_"+scenario_id}...')
            
            # create one-hot vector
            one_hot_id = speaker_to_id[actor]
            id = torch.zeros([1, cfg.id_dim])
            id[0, one_hot_id] = 1
            
            # load audio
            audio = load_example_input(os.path.join(WAV_PATH, date_id+'_'+actor+'_'+scenario_id+'.wav'))
            
            # make prediction
            data_input = {
                'audio': audio.to(device),
                'id': id.to(device)
            }
            with torch.no_grad():
                prediction = model.predict(data_input)
                vertices = prediction['vertice_pred'].squeeze().cpu().numpy()
        
            # save outputs
            pred_file = os.path.join(PRED_PATH, date_id+'_'+actor+'_'+scenario_id+'.npy')
            np.save(pred_file, vertices)
    
    print('All done!')


if __name__ == "__main__":
    main()