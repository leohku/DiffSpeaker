export CUDA_VISIBLE_DEVICES=0

    # --example_male /data3/leoho/arfriend-diffspeaker/wav/20231119_002Shirley_053.wav \
    # --example_female /data3/leoho/arfriend-diffspeaker/wav/20231119_001Sky_053.wav \
    # --example_male demo/wavs/speech_obama.wav \
    # --example_female demo/wavs/speech_obama.wav \ 

python demo_arfriend3.py \
    --cfg configs/diffusion/arfriend3/diffspeaker_wav2vec2_arfriend3.yaml \
    --cfg_assets configs/assets/arfriend3.yaml \
    --template datasets/arfriend/templates.pkl \
    --example_male /data3/leoho/arfriend-diffspeaker/wav/20231119_001Sky_053.wav \
    --example_female /data3/leoho/arfriend-diffspeaker/wav/20231119_002Shirley_053.wav \
    --angle 0 \
    --distance 0.8\
    --ply datasets/arfriend/templates/001Sky.obj \
    --checkpoint experiments/arfriend3/diffusion_bias/diffspeaker_wav2vec2_arfriend3/checkpoints/epoch=64.ckpt \
    --id "20240126"