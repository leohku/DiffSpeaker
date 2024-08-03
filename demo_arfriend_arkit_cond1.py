import os
import pickle
import numpy as np
import torch

from alm.config import parse_args
from alm.models.get_model import get_model
from alm.utils.logger import create_logger
from alm.utils.demo_utils import animate

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
    dataset = 'arfriend_arkit_cond1' # TODO
    logger = create_logger(cfg, phase="demo")

    # set up the model architecture
    cfg.DATASET.NFEATS = 55
    model = get_model(cfg, dataset)

    if cfg.DEMO.EXAMPLE:
        # load audio input 
        logger.info("Loading audio from {}".format(cfg.DEMO.EXAMPLE))
        from alm.utils.demo_utils import load_example_input
        audio_path = cfg.DEMO.EXAMPLE
        assert os.path.exists(audio_path), 'audio does not exist'
        audio = load_example_input(audio_path)
    else:
        raise NotImplemented
    
    if cfg.DEMO.EXAMPLE_AUDIO_COND:
        # load audio condition
        logger.info("Loading audio condition from {}".format(cfg.DEMO.EXAMPLE_AUDIO_COND))
        from alm.utils.demo_utils import load_example_input
        audio_cond_path = cfg.DEMO.EXAMPLE_AUDIO_COND
        assert os.path.exists(audio_cond_path), 'audio condition does not exist'
        audio_cond = load_example_input(audio_cond_path)
    else:
        raise NotImplemented
    
    if cfg.DEMO.EXAMPLE_VERTICE_COND:
        vertice_cond = torch.FloatTensor(np.load(cfg.DEMO.EXAMPLE_VERTICE_COND))
    else:
        raise NotImplemented

    # truncate or extend audio_cond if size of audio and audio_cond is different
    if audio.shape != audio_cond.shape:
        if audio.shape[0] > audio_cond.shape[0]:
            audio_cond = np.pad(audio_cond, (0, audio.shape[0] - audio_cond.shape[0]), 'constant', constant_values=0)
        else:
            audio_cond = audio_cond[:audio.shape[0]]

    # load model weights
    logger.info("Loading checkpoints from {}".format(cfg.DEMO.CHECKPOINTS))
    state_dict = torch.load(cfg.DEMO.CHECKPOINTS, map_location="cpu")["state_dict"]

    state_dict.pop("denoiser.PPE.pe") # this is not needed, since the sequence length can be any flexiable
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # paraterize the speaking style
    speaker_to_id = {
        '001Sky': 0,
        '002Shirley': 1,
        '003Alan': 2,
        '005Richard': 3,
        '006Vasilisa': 4,
        '007Jessica': 5,
        '008Kunio': 6,
    }
    
    if cfg.DEMO.ID in speaker_to_id:
        speaker_id = speaker_to_id[cfg.DEMO.ID]
        id = torch.zeros([1, cfg.id_dim])
        id[0, speaker_id] = 1
    else:
        id = torch.zeros([1, cfg.id_dim])
        id[0, 0] = 1

    # make prediction
    logger.info("Making predictions")
    data_input = {
        'audio': audio.to(device),
        'audio_cond': audio_cond.to(device),
        'vertice_cond': vertice_cond.to(device),
        'id': id.to(device),
    }
    with torch.no_grad():
        prediction = model.predict(data_input)
        vertices = prediction['vertice_pred'].squeeze().cpu().numpy()

    # this function is copy from faceformer
    wav_path = cfg.DEMO.EXAMPLE
    test_name = os.path.basename(wav_path).split(".")[0]
    
    subject_id = cfg.DEMO.ID
    output_base_dir = os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME))
    output_dir = os.path.join(output_base_dir, "samples_" + cfg.TIME)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = os.path.join(output_dir,test_name + "_" + subject_id + '.npy')

    np.save(file_name, vertices)

if __name__ == "__main__":
    main()