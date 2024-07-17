export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m train_arkit \
    --cfg configs/diffusion/arfriend_arkit/diffspeaker_wav2vec2_arfriend_arkit_fine_base.yaml \
    --cfg_assets configs/assets/arfriend_arkit.yaml \
    --batch_size 2 \
    --nodebug \
