#!/bin/sh

python -m exp.run \
    --dataset=cora\
    --d=4 \
    --k=1 \
    --sample_budget=2\
    --layers=4 \
    --hidden_channels=20 \
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
    