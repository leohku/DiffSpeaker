export CUDA_VISIBLE_DEVICES=0

# --example datasets/lipacc/wav/20240504_006Vasilisa_046.wav \
# --example demo/wavs/chemistry.wav \
# --example demo/wavs/speech_british.wav \
# --example demo/wavs/speech_obama.wav \

EPOCH_NUMBER=4799

python demo_arfriend_arkit.py \
    --cfg configs/diffusion/arfriend_arkit/diffspeaker_wav2vec2_arfriend_arkit_fine_base.yaml \
    --cfg_assets configs/assets/arfriend_arkit.yaml \
    --example datasets/lipacc/wav/20240504_006Vasilisa_046.wav \
    --checkpoint experiments/arfriend_arkit/diffusion_bias_arkit/diffspeaker_wav2vec2_arfriend_arkit_fine_base/checkpoints/epoch=${EPOCH_NUMBER}.ckpt \
    --id 006Vasilisa
