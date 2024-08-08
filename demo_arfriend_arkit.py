import os
import pickle
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
    dataset = 'arfriend_arkit' # TODO
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

    # load model weights
    logger.info("Loading checkpoints from {}".format(cfg.DEMO.CHECKPOINTS))
    state_dict = torch.load(cfg.DEMO.CHECKPOINTS, map_location="cpu")["state_dict"]
    state_dict.pop("denoiser.PPE.pe") # this is not needed, since the sequence length can be any flexiable
    state_dict_copy = {}    
    
    for key in state_dict.keys():
        state_dict_copy[key] = state_dict[key]

    # botched
    # lipacc_state_dict = torch.load("/home/leoho/diffspeaker/experiments/arfriend_arkit_cond1/diffusion_bias_arkit_cond1/diffspeaker_wav2vec2_arfriend_arkit_cond2/checkpoints/epoch=599.ckpt", map_location="cpu")["state_dict"]
    
    # lipacc
    # lipacc_state_dict = torch.load("/home/leoho/diffspeaker/experiments/lipacc_arkit/diffusion_bias_arkit/diffspeaker_wav2vec2_lipacc_arkit_hyper/checkpoints/epoch=4799.ckpt", map_location="cpu")["state_dict"]
    
    # fine tuned
    # lipacc_state_dict = torch.load("/home/leoho/diffspeaker/experiments/lipacc_arkit/diffusion_bias_arkit/diffspeaker_wav2vec2_lipacc_arkit_fine_tune_fixid_no_face_loss/checkpoints/epoch=3434.ckpt", map_location="cpu")["state_dict"]
    # lipacc_state_dict = torch.load("/home/leoho/diffspeaker/experiments/lipacc_arkit/diffusion_bias_arkit/diffspeaker_wav2vec2_lipacc_arkit_fine_tune_fixid/checkpoints/epoch=3449.ckpt", map_location="cpu")["state_dict"]
    # for key in lipacc_state_dict.keys():
    #     if "transformer_decoder" in key:
    #         # replace "transformer_decoder" with "lipacc_transformer_decoder
    #         key2 = key.replace("transformer_decoder", "lipacc_transformer_decoder")
    #         state_dict_copy[key2] = lipacc_state_dict[key]
            
    state_dict = state_dict_copy
    
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