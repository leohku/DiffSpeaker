export CUDA_VISIBLE_DEVICES=0

# python demo_arfriend2.py \
#     --cfg configs/diffusion/arfriend2/diffspeaker_wav2vec2_arfriend2.yaml \
#     --cfg_assets configs/assets/arfriend2.yaml \
#     --template datasets/arfriend/templates.pkl \
#     --example_male demo/wavs/speech_obama.wav \
#     --example_female demo/wavs/speech_obama.wav \
#     --ply datasets/arfriend/templates/001Sky.obj \
#     --checkpoint experiments/arfriend2/diffusion_bias/diffspeaker_wav2vec2_arfriend2/checkpoints/epoch=349.ckpt \
#     --id "20231119"

python demo_arfriend2.py \
    --cfg configs/diffusion/arfriend2/diffspeaker_wav2vec2_arfriend2.yaml \
    --cfg_assets configs/assets/arfriend2.yaml \
    --template datasets/arfriend/templates.pkl \
    --example_male /data3/leoho/arfriend-diffspeaker/wav/20231119_002Shirley_053.wav \
    --example_female /data3/leoho/arfriend-diffspeaker/wav/20231119_001Sky_053.wav \
    --ply datasets/arfriend/templates/001Sky.obj \
    --checkpoint experiments/arfriend2/diffusion_bias/diffspeaker_wav2vec2_arfriend2/checkpoints/epoch=349.ckpt \
    --id "20240126"