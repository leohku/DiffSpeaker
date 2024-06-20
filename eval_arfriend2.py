import os
import pickle
import torch

from alm.config import parse_args
from alm.models.get_model import get_model
from alm.utils.logger import create_logger
from alm.utils.demo_utils import animate, load_example_input

import numpy as np

MODEL_CKPT = '/home/leoho/diffspeaker/checkpoints/arfriend2/epoch=359.ckpt'
EVAL_LIST = '/data3/leoho/arfriend-diffspeaker/test_list_remainder.txt'
TEMPLATE_FILE = '/data3/leoho/arfriend-diffspeaker/templates.pkl'
WAV_PATH = '/data3/leoho/arfriend-diffspeaker/wav'
PRED_PATH = '/home/leoho/diffspeaker/evals/arfriend2-test-set'

# template info
date_order = {
    "20231119": 0,
    "20231126": 1,
    "20231208": 2,
    "20240126": 3,
    "20240128": 4
}
date_subjects = {
    "20231119": ["001Sky", "002Shirley"],
    "20231126": ["003Alan", "007Jessica"],
    "20231208": ["005Richard", "006Vasilisa"],
    "20240126": ["008Kunio", "006Vasilisa"],
    "20240128": ["001Sky", "007Jessica"]
}

def main():
    cfg = parse_args(phase="demo")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")
    
    dataset = 'arfriend2'
    logger = create_logger(cfg, phase="demo")
    # set up the model architecture
    cfg.DATASET.NFEATS = 72147 * 2
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
    
    # load template file
    with open(TEMPLATE_FILE, 'rb') as f:
        templates = pickle.load(f, encoding='latin1')
    
    for eval in eval_targets:
        print(f'Predicting {eval}...')
        
        date_id = eval.split('_')[0]
        scenario_id = eval.split('_')[1]

        # create one-hot vector
        one_hot_id = date_order[date_id]
        id = torch.zeros([1, cfg.id_dim])
        id[0, one_hot_id] = 1
        
        # get templates
        actor_list = date_subjects[date_id]
        template_male = templates[actor_list[0]].reshape(-1)
        template_female = templates[actor_list[1]].reshape(-1)
        template = torch.Tensor(np.concatenate((template_male, template_female)))
        
        # load audio
        audio_male = load_example_input(os.path.join(WAV_PATH, date_id+'_'+actor_list[0]+'_'+scenario_id+'.wav'))
        audio_female = load_example_input(os.path.join(WAV_PATH, date_id+'_'+actor_list[1]+'_'+scenario_id+'.wav'))
        # fix for uneven audio lengths
        if audio_male.shape[1] != audio_female.shape[1]:
            length = min(audio_male.shape[1], audio_female.shape[1])
            audio_male = audio_male[:, :length]
            audio_female = audio_female[:, :length]
        
        # make prediction
        data_input = {
            'audio_male': audio_male.to(device),
            'audio_female': audio_female.to(device),
            'template': template.to(device),
            'id': id.to(device)
        }
        with torch.no_grad():
            prediction = model.predict(data_input)
            vertices = prediction['vertice_pred'].squeeze().cpu().numpy()
        
        # save outputs
        pred_male_file = os.path.join(PRED_PATH, date_id+'_'+actor_list[0]+'_'+scenario_id+'.npy')
        pred_female_file = os.path.join(PRED_PATH, date_id+'_'+actor_list[1]+'_'+scenario_id+'.npy')
        
        np.save(pred_male_file, vertices[:,:(cfg.DATASET.NFEATS // 2)])
        np.save(pred_female_file, vertices[:,(cfg.DATASET.NFEATS // 2):])
    
    print('All done!')


if __name__ == "__main__":
    main()