#!/bin/sh

python -m exp.run \
    --dataset=texas\
    --d=128 \
    --k=2     \
    --sampler=uniform\
    --sample_budget=1\
    --epochs=100 \
    --layers=4 \
    --hidden_channels=8 \
    --left_weights=True \
    --right_weights=True \
    --lr=0.02 \
    --weight_decay=5e-3 \
    --input_dropout=0.0 \
    --dropout=0.7 \
    --use_act=True \
    --model=SampleBundleSheaf \
    --normalised=True \
    --sparse_learner=True \
    --entity="${ENTITY}"
