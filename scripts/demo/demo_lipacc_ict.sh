export CUDA_VISIBLE_DEVICES=0

# --example datasets/lipacc/wav/20240504_006Vasilisa_046.wav \
# --example demo/wavs/chemistry.wav \
# --example demo/wavs/speech_british.wav \
# --example demo/wavs/speech_obama.wav \

EPOCH_NUMBER=1999

python demo_lipacc_ict.py \
    --cfg configs/diffusion/lipacc_ict/diffspeaker_wav2vec2_lipacc_ict.yaml \
    --cfg_assets configs/assets/lipacc_ict.yaml \
    --template datasets/lipacc/templates_ict.pkl \
    --example datasets/lipacc/wav/20240504_006Vasilisa_046.wav \
    --ply datasets/lipacc/templates_ict/006Vasilisa.obj \
    --checkpoint experiments/lipacc_ict/diffusion_bias/diffspeaker_wav2vec2_lipacc_ict/checkpoints/epoch=${EPOCH_NUMBER}.ckpt \
    --id 006Vasilisa

# python demo_lipacc_ict.py \
#     --cfg configs/diffusion/lipacc_ict/diffspeaker_wav2vec2_lipacc_ict.yaml \
#     --cfg_assets configs/assets/lipacc_ict.yaml \
#     --template datasets/lipacc/templates_ict.pkl \
#     --example demo/wavs/chemistry.wav \
#     --ply datasets/lipacc/templates_ict/006Vasilisa.obj \
#     --checkpoint experiments/lipacc_ict/diffusion_bias/diffspeaker_wav2vec2_lipacc_ict/checkpoints/epoch=${EPOCH_NUMBER}.ckpt \
#     --id 006Vasilisa

# python demo_lipacc_ict.py \
#     --cfg configs/diffusion/lipacc_ict/diffspeaker_wav2vec2_lipacc_ict.yaml \
#     --cfg_assets configs/assets/lipacc_ict.yaml \
#     --template datasets/lipacc/templates_ict.pkl \
#     --example demo/wavs/speech_british.wav \
#     --ply datasets/lipacc/templates_ict/006Vasilisa.obj \
#     --checkpoint experiments/lipacc_ict/diffusion_bias/diffspeaker_wav2vec2_lipacc_ict/checkpoints/epoch=${EPOCH_NUMBER}.ckpt \
#     --id 006Vasilisa

# python demo_lipacc_ict.py \
#     --cfg configs/diffusion/lipacc_ict/diffspeaker_wav2vec2_lipacc_ict.yaml \
#     --cfg_assets configs/assets/lipacc_ict.yaml \
#     --template datasets/lipacc/templates_ict.pkl \
#     --example demo/wavs/speech_obama.wav \
#     --ply datasets/lipacc/templates_ict/006Vasilisa.obj \
#     --checkpoint experiments/lipacc_ict/diffusion_bias/diffspeaker_wav2vec2_lipacc_ict/checkpoints/epoch=${EPOCH_NUMBER}.ckpt \
#     --id 006Vasilisa