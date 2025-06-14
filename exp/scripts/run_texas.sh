#!/bin/sh

python -m exp.run \
    --dataset=cora\
    --d=4 \
    --sampler=uniform\
    --sample_budget=1\
    --epochs=500 \
    --layers=2 \
    --hidden_channels=32 \
    --left_weights=True \
    --right_weights=True \
    --lr=0.01 \
    --weight_decay=5e-3 \
    --input_dropout=0.0 \
    --dropout=0.7 \
    --use_act=True \
    --model=BundleSheaf \
    --normalised=True \
    --sparse_learner=True \
    --entity="${ENTITY}"
