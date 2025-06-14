SWEEP_ID=$(wandb sweep exp/sweep.yaml)
echo "Sweep ID is $SWEEP_ID"
wandb agent $SWEEP_ID --count 280
