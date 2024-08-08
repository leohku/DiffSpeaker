export CUDA_VISIBLE_DEVICES=0

# --example datasets/lipacc/wav/20240504_006Vasilisa_046.wav \
# --example demo/wavs/chemistry.wav \
# --example demo/wavs/speech_british.wav \
# --example demo/wavs/speech_obama.wav \
# --example datasets/arfriend/wav/20231126_007Jessica_310.wav \

EPOCH_NUMBER=4799

python demo_lipacc_arkit.py \
    --cfg configs/diffusion/lipacc_arkit/diffspeaker_wav2vec2_lipacc_arkit_hyper.yaml \
    --cfg_assets configs/assets/lipacc_arkit.yaml \
    --example /data3/leoho/arfriend-diffspeaker/wav/20231126_003Alan_211.wav \
    --checkpoint experiments/lipacc_arkit/diffusion_bias_arkit/diffspeaker_wav2vec2_lipacc_arkit_hyper/checkpoints/epoch=${EPOCH_NUMBER}.ckpt \
    --id 006Vasilisa
