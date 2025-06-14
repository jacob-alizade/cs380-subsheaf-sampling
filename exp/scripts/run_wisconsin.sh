#!/bin/sh

python -m exp.run \
    --add_hp=True \
    --add_lp=True \
    --d=6 \
    --dataset=texas \
    --dropout=0.7 \
    --early_stopping=100 \
    --input_dropout=0.7\
    --epochs=200 \
    --folds=10 \
    --hidden_channels=16 \
    --layers=4 \
    --lr=0.01 \
    --model=BundleSheaf \
    --orth=householder \
    --sheaf_act=elu \
    --weight_decay=0.0005 \
    --use_act=True \
    --normalised=True \
    --sparse_learner=True \
    --edge_weights=True \
    --entity="${ENTITY}"