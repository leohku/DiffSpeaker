export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m train \
    --cfg configs/diffusion/arfriend/diffspeaker_wav2vec2_arfriend.yaml \
    --cfg_assets configs/assets/arfriend.yaml \
    --batch_size 32 \
    --nodebug \
