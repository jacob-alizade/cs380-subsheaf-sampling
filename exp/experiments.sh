SWEEP_ID=$(wandb sweep sweep.yaml)
echo "Sweep ID is $SWEEP_ID"
wandb agent $SWEEP_ID --count 150
