export CUDA_VISIBLE_DEVICES=0

# --example datasets/lipacc/wav/20240504_006Vasilisa_046.wav \
# --example demo/wavs/chemistry.wav \
# --example demo/wavs/speech_british.wav \
# --example demo/wavs/speech_obama.wav \
# --example /data3/leoho/arfriend-diffspeaker/wav/20231119_002Shirley_769.wav \

EPOCH_NUMBER=3449

python demo_lipacc_arkit.py \
    --cfg configs/diffusion/lipacc_arkit/diffspeaker_wav2vec2_lipacc_arkit_fine_tune_no_face_loss.yaml \
    --cfg_assets configs/assets/lipacc_arkit.yaml \
    --example /home/leoho/faceformer/demo/wav/semester.wav \
    --checkpoint experiments/lipacc_arkit/diffusion_bias_arkit/diffspeaker_wav2vec2_lipacc_arkit_fine_tune_no_face_loss/checkpoints/epoch=${EPOCH_NUMBER}.ckpt \
    --id 002Shirley
    