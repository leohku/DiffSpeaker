export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m train \
    --cfg configs/diffusion/arfriend2/diffspeaker_wav2vec2_arfriend2.yaml \
    --cfg_assets configs/assets/arfriend2.yaml \
    --batch_size 4 \
    --nodebug \
