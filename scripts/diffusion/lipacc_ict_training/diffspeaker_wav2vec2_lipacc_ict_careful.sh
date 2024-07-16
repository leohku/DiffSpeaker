export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m train \
    --cfg configs/diffusion/lipacc_ict/diffspeaker_wav2vec2_lipacc_ict_careful.yaml \
    --cfg_assets configs/assets/lipacc_ict.yaml \
    --batch_size 2 \
    --nodebug \
