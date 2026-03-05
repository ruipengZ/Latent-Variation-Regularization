python examples/imitation/go2_imitation_LVR.py \
  --task "Isaac-Velocity-Flat-Forward-Unitree-Go2-v0" \
  --dataset-path 'imitation_data/isaac_go2_forward/traj-1.hdf5' \
  --k-neighbors 64 \
  --anchors-per-update 32 \
  --epoch 100

python examples/imitation/go2_imitation_LVR.py \
  --task "Isaac-Velocity-Flat-Sideway-Unitree-Go2-v0" \
  --dataset-path 'imitation_data/isaac_go2_sideway/traj-1.hdf5' \
  --k-neighbors 32 \
  --anchors-per-update 32 \
  --epoch 100

python examples/imitation/go2_imitation_LVR.py \
  --task "Isaac-Velocity-Flat-Backward-Unitree-Go2-v0" \
  --dataset-path 'imitation_data/isaac_go2_backward/traj-1.hdf5' \
  --k-neighbors 32 \
  --anchors-per-update 32 \
  --epoch 100