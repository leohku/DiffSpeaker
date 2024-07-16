export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m train \
    --cfg configs/diffusion/lipacc_arkit/diffspeaker_wav2vec2_lipacc_arkit_hyper_layerx2.yaml \
    --cfg_assets configs/assets/lipacc_arkit.yaml \
    --batch_size 2 \
    --nodebug \
