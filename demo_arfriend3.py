import os
import pickle
import torch

from alm.config import parse_args
from alm.models.get_model import get_model
from alm.utils.logger import create_logger
from alm.utils.demo_utils import animate, load_example_input

import math
import numpy as np

def main():
    # parse options
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME

    # set up the device
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in cfg.DEVICE)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # set up the logger
    dataset = 'arfriend3' # TODO
    logger = create_logger(cfg, phase="demo")

    # set up the model architecture
    cfg.DATASET.NFEATS = 72147 * 2
    model = get_model(cfg, dataset)

    assert cfg.DEMO.EXAMPLE_MALE is not None and cfg.DEMO.EXAMPLE_FEMALE is not None
    # load audio inputs
    assert os.path.exists(cfg.DEMO.EXAMPLE_MALE), 'male audio does not exist'
    assert os.path.exists(cfg.DEMO.EXAMPLE_FEMALE), 'female audio does not exist'
    logger.info("Loading male audio from {}".format(cfg.DEMO.EXAMPLE_MALE))
    audio_male = load_example_input(cfg.DEMO.EXAMPLE_MALE)
    logger.info("Loading female audio from {}".format(cfg.DEMO.EXAMPLE_FEMALE))
    audio_female = load_example_input(cfg.DEMO.EXAMPLE_FEMALE)

    # load model weights
    logger.info("Loading checkpoints from {}".format(cfg.DEMO.CHECKPOINTS))
    state_dict = torch.load(cfg.DEMO.CHECKPOINTS, map_location="cpu")["state_dict"]

    state_dict.pop("denoiser.PPE.pe") # this is not needed, since the sequence length can be any flexiable
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
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

    # load the template
    assert cfg.DEMO.ID in date_order and cfg.DEMO.ID in date_subjects
    logger.info("Loading template mesh from {}".format(cfg.DEMO.TEMPLATE))
    template_file = cfg.DEMO.TEMPLATE
    with open(template_file, 'rb') as fin:
        template = pickle.load(fin,encoding='latin1')
    date_id = cfg.DEMO.ID
    template_male = template[date_subjects[date_id][0]].reshape(-1)
    template_female = template[date_subjects[date_id][1]].reshape(-1)
    template = torch.Tensor(np.concatenate((template_male, template_female)))
    
    # make one-hot identity vector
    one_hot_id = date_order[cfg.DEMO.ID]
    id = torch.zeros([1, cfg.id_dim])
    id[0, one_hot_id] = 1

    # make relative tensor
    # angle_rad = float(cfg.DEMO.ANGLE) * (np.pi / 180)
    # distance_m = float(cfg.DEMO.DISTANCE)
    # rel = torch.FloatTensor([math.sin(angle_rad), math.cos(angle_rad), distance_m]).unsqueeze(0)

    rel = torch.FloatTensor([1,0] if cfg.DEMO.FACING == "yes" else [0,1]).unsqueeze(0)
    print('Relative vector: ', rel)

    # make prediction
    logger.info("Making predictions")
    data_input = {
        'audio_male': audio_male.to(device),
        'audio_female': audio_female.to(device),
        'rel': rel.to(device),
        'template': template.to(device),
        'id': id.to(device),
    }
    with torch.no_grad():
        prediction = model.predict(data_input)
        vertices = prediction['vertice_pred'].squeeze().cpu().numpy()

    # this function is copy from faceformer
    output_dir = os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME), "samples_" + cfg.TIME)
    
    # male output
    wav_path = cfg.DEMO.EXAMPLE_MALE
    test_name = os.path.basename(wav_path).split(".")[0]
    file_name_male = os.path.join(output_dir, test_name + "_" + date_id + '_male.mp4')
    animate(vertices[:,:(cfg.DATASET.NFEATS // 2)], wav_path, file_name_male, cfg.DEMO.PLY, fps=30, use_tqdm=True, multi_process=True)
    
    # female output
    wav_path = cfg.DEMO.EXAMPLE_FEMALE
    test_name = os.path.basename(wav_path).split(".")[0]
    file_name_female = os.path.join(output_dir, test_name + "_" + date_id + '_female.mp4')
    animate(vertices[:,(cfg.DATASET.NFEATS // 2):], wav_path, file_name_female, cfg.DEMO.PLY, fps=30, use_tqdm=True, multi_process=True)
    
    # merge two outputs in one video
    file_name_both = os.path.join(output_dir, test_name + "_" + date_id + '_both.mp4')
    os.system(f'ffmpeg -i {file_name_male} -i {file_name_female} -filter_complex "[0:v][1:v]hstack=inputs=2[v]; [0:a][1:a]amix=inputs=2[a]" -map "[v]" -map "[a]" {file_name_both}')

if __name__ == "__main__":
    main()