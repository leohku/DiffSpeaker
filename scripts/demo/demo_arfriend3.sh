export CUDA_VISIBLE_DEVICES=0

    # --example_male /data3/leoho/arfriend-diffspeaker/wav/20231119_002Shirley_053.wav \
    # --example_female /data3/leoho/arfriend-diffspeaker/wav/20231119_001Sky_053.wav \
    # --example_male demo/wavs/speech_obama.wav \
    # --example_female demo/wavs/speech_obama.wav \
    # --example_male /data3/leoho/arfriend-diffspeaker/wav/20240126_008Kunio_034.wav \
    # --example_female /data3/leoho/arfriend-diffspeaker/wav/20240126_006Vasilisa_034.wav \

    # --angle 30 \
    # --distance 0.8\

python demo_arfriend3.py \
    --cfg configs/diffusion/arfriend3/diffspeaker_wav2vec2_arfriend3.yaml \
    --cfg_assets configs/assets/arfriend3.yaml \
    --template datasets/arfriend/templates.pkl \
    --example_male demo/wavs/speech_obama.wav \
    --example_female demo/wavs/speech_obama.wav \
    --facing no \
    --ply datasets/arfriend/templates/001Sky.obj \
    --checkpoint experiments/arfriend3/diffusion_bias/diffspeaker_wav2vec2_arfriend3/checkpoints/epoch=224-v1.ckpt \
    --id "20240126"