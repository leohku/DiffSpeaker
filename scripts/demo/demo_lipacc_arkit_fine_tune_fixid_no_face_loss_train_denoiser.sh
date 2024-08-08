export CUDA_VISIBLE_DEVICES=4

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

EPOCH_NUMBER=3434

python demo_arfriend_arkit.py \
    --cfg configs/diffusion/lipacc_arkit/diffspeaker_wav2vec2_lipacc_arkit_fine_tune_fixid_no_face_loss_train_denoiser.yaml \
    --cfg_assets configs/assets/lipacc_arkit.yaml \
    --example /home/leoho/faceformer/demo/wav/20231119_002Shirley_052.wav \
    --checkpoint experiments/lipacc_arkit/diffusion_bias_arkit/diffspeaker_wav2vec2_lipacc_arkit_fine_tune_fixid_no_face_loss_train_denoiser/checkpoints/epoch=${EPOCH_NUMBER}.ckpt \
    --id 002Shirley
    