### Dataset Recording Commands 

```sh
# Terminal 1 
cd /Development/openpi
source recording_env/bin./activate
python scripts/lerobot_record_openarm_openpi.py \
    --repo_id=saurabh/openarm_pick_v5 \
    --task="grab the electric screwdriver from the table and drop it inside the box" \
    --num_episodes=100
```

```sh
# terminal 2 (teloperation)
cd /Development/quest3_streamer
./scripts/run_openarm_teleop.sh

```


### Training/Finetuning Commands

```sh
# Step 1: Generate normalization statistics (run once per new dataset)
cd Development/openpi
uv run python scripts/compute_openarm_norm_stats.py
# Output: assets/pi05_openarm/saurabh/openarm_pick_v6/
```

```sh
# training from scratch
cd Development/openpi
source .venv/bin/activate
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_openarm \
    --exp-name=openarm_test_3k \
    --overwrite

#training continue from last checkpoint 
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_openarm \
    --exp-name=openarm_test_3k \
    --resume

```

### Inference Commands

```sh 
#terminal 1 
cd Development/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_openarm \
    --policy.dir=./checkpoints/pi05_openarm/openarm_pi0.5_finetuned/5000
```

```sh
#terminal 2 
cd Development/openpi
source /opt/ros/humble/setup.bash
source client_env/bin/activate
python scripts/openarm_policy_client.py \
    --task="grab the electric screwdriver from the table and drop it inside the box"\
    --fps=30.0
```

```sh
#terminal 3 
cd Development/lerobot/
./src/lerobot/scripts/run_ros_bridge.sh
```

### Hugging Face Upload Commands

```sh
# Upload checkpoint to Hugging Face
cd Development/openpi
source .venv/bin/activate
python scripts/upload_checkpoint_to_hf.py \
    --checkpoint-dir=./checkpoints/pi05_openarm/openarm_pi0.5_finetuned/5000 \
    --repo-name=openarm-pi05-finetuned

# Upload dataset to Hugging Face  
python scripts/upload_dataset_to_hf.py \
    --dataset-path=~/.cache/huggingface/lerobot/saurabh/openarm_pick_v6 \
    --repo-name=openarm_pick_v6

# Dry run (preview without uploading)
python scripts/upload_dataset_to_hf.py --dry-run
python scripts/upload_checkpoint_to_hf.py --dry-run
```
