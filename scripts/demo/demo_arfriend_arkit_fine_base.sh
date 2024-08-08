export CUDA_VISIBLE_DEVICES=0

# --example datasets/lipacc/wav/20240504_006Vasilisa_046.wav \
# --example demo/wavs/chemistry.wav \
# --example demo/wavs/speech_british.wav \
# --example demo/wavs/speech_obama.wav \
# --example /data3/leoho/arfriend-diffspeaker/wav/20231119_002Shirley_769.wav \
# --example /home/leoho/faceformer/demo/wav/justine-4x.wav \
# --example /home/leoho/faceformer/demo/wav/semester.wav \
# --example /home/leoho/faceformer/demo/wav/lecture-2x.wav \
# --example /home/leoho/faceformer/demo/wav/20231119_002Shirley_052.wav \
# --example /data3/leoho/arfriend-diffspeaker/wav/20231126_003Alan_211.wav \
# --example /data3/leoho/arfriend-diffspeaker/wav/20231126_007Jessica_210.wav \
# --example /data3/leoho/arfriend-diffspeaker/wav/20231208_005Richard_158.wav \
# --example /data3/leoho/arfriend-diffspeaker/wav/20231119_001Sky_068.wav \

EPOCH_NUMBER=3429

python demo_arfriend_arkit.py \
    --cfg configs/diffusion/arfriend_arkit/diffspeaker_wav2vec2_arfriend_arkit_fine_base.yaml \
    --cfg_assets configs/assets/arfriend_arkit.yaml \
    --example /data3/leoho/arfriend-diffspeaker/wav/20231119_001Sky_068.wav \
    --checkpoint experiments/arfriend_arkit/diffusion_bias_arkit/diffspeaker_wav2vec2_arfriend_arkit_fine_base_cont/checkpoints/epoch=${EPOCH_NUMBER}.ckpt \
    --id 001Sky
