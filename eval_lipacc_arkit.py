import os
import torch

from alm.config import parse_args
from alm.models.get_model import get_model
from alm.utils.logger import create_logger
from alm.utils.demo_utils import load_example_input

import numpy as np

MODEL_CKPT = '/home/leoho/diffspeaker/checkpoints/lipacc_arkit/epoch=359.ckpt'
EVAL_LIST = '/home/leoho/diffspeaker/datasets/lipacc/test_list.txt'
WAV_PATH = '/home/leoho/diffspeaker/datasets/lipacc/wav'
PRED_PATH = '/home/leoho/diffspeaker/evals/lipacc_arkit-test-set'

# paraterize the speaking style
speaker_to_id = {
    "006Vasilisa": 0
}
date_subjects = {
    "20240504": ["006Vasilisa"]
}

def main():
    cfg = parse_args(phase="demo")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")
    
    dataset = 'lipacc_arkit'
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
        print(f'Predicting {eval}...')
        
        date_id = eval.split('_')[0]
        scenario_id = eval.split('_')[1]
        
        for actor in date_subjects[date_id]:           
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