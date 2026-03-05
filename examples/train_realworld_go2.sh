python examples/imitation/go2_imitation_LVR.py \
  --task "Unitree-Go2-Velocity-Forward" \
  --dataset-path 'imitation_data/unitree_go2_forward/step-250+250.hdf5' \
  --k-neighbors 64 \
  --anchors-per-update 32 \
  --epoch 100

python examples/imitation/go2_imitation_LVR.py \
  --task "Unitree-Go2-Velocity-Sideway" \
  --dataset-path 'imitation_data/unitree_go2_sideway/step-250+250.hdf5' \
  --k-neighbors 64 \
  --anchors-per-update 32 \
  --epoch 100

python examples/imitation/go2_imitation_LVR.py \
  --task "Unitree-Go2-Velocity-Backward" \
  --dataset-path 'imitation_data/unitree_go2_backward/step-250+250.hdf5' \
  --k-neighbors 64 \
  --anchors-per-update 32 \
  --epoch 100