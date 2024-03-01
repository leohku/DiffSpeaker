export CUDA_VISIBLE_DEVICES=0

python demo_arfriend.py \
    --cfg configs/diffusion/arfriend/diffspeaker_wav2vec2_arfriend.yaml \
    --cfg_assets configs/assets/arfriend.yaml \
    --template datasets/arfriend/templates.pkl \
    --example /data3/leoho/arfriend-diffspeaker/wav/20231119_002Shirley_053.wav \
    --ply datasets/arfriend/templates/001Sky.obj \
    --checkpoint experiments/arfriend/diffusion_bias/diffspeaker_wav2vec2_arfriend/checkpoints/epoch=86.ckpt \
    --id 002Shirley