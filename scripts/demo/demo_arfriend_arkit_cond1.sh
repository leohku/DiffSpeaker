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

EPOCH_NUMBER=899

python demo_arfriend_arkit_cond1.py \
    --cfg configs/diffusion/arfriend_arkit_cond1/diffspeaker_wav2vec2_arfriend_arkit_cond1.yaml \
    --cfg_assets configs/assets/arfriend_arkit_cond1.yaml \
    --example /data3/leoho/arfriend-diffspeaker/wav/20231126_003Alan_211.wav \
    --example_audio_cond /data3/leoho/arfriend-diffspeaker/wav/20231126_007Jessica_211.wav \
    --example_vertice_cond /data3/leoho/arfriend-diffspeaker/arkit_npy/20231126_007Jessica_211.npy \
    --checkpoint experiments/arfriend_arkit_cond1/diffusion_bias_arkit_cond1/diffspeaker_wav2vec2_arfriend_arkit_cond1/checkpoints/epoch=${EPOCH_NUMBER}.ckpt \
    --id 003Alan
