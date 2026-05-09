# TetraxCode/openarm_pick_place_v1 recon

Snapshot inspected:
`/mnt/nas/pick/huggingface/hub/datasets--TetraxCode--openarm_pick_place_v1/snapshots/ffa74d1fa362b3df62e30adb2810688e2b9a54e9`

Notes:
- Initial `uv run huggingface-cli download ... --include "data/chunk-000/episode_000000.parquet"` succeeded but fetched no data parquet because this is a LeRobot v3 dataset whose `meta/info.json` declares `data_path: data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`.
- The first data parquet inspected was therefore `data/chunk-000/file-000.parquet`.
- LeRobot metadata loads with a warning that the dataset is in 3.0 format and uses global stats.

| Item | Filled value | Branch / implication |
|---|---|---|
| `features.observation.state.shape` | `(16,)`, dtype `float32`; first parquet materializes as `(7402, 16)`. | H1 on dimension: current 16-D config can consume the shape. |
| `features.action.shape` | `(16,)`, dtype `float32`; first parquet materializes as `(7402, 16)`. | H1 on dimension: current 16-D config can consume the shape. |
| `state.names` (joint order) | `joint_0` ... `joint_15`; action names are `action_joint_0` ... `action_joint_15`. The file is 16-D pos-only, but metadata does not expose semantic names like `left_joint_1` or `right_gripper`. Existing config/contract assumes indices `0..7` are left arm + left gripper and `8..15` are right arm + right gripper. | Dimension and pos-only layout match `STATE_ORDER`; semantic left/right order cannot be proven from dataset names alone. |
| Camera column names and mapping | Metadata exposes exactly `observation.images.cam_0`, `observation.images.cam_2`, and `observation.images.cam_4`, each video `(480, 640, 3)`, fps 30, RGB-like video, not depth. `cam_1` and `cam_3` are absent. Visual sample check: `cam_0` is overhead/head view; `cam_2` and `cam_4` are wrist/gripper views. Existing config maps `head <- cam_0`, `wrist_left <- cam_2`, `wrist_right <- cam_4`. | Column names match config. Metadata has no explicit left/right wrist labels, so handedness follows the existing config mapping. |
| `tasks.jsonl` existence + first prompt | `meta/tasks.jsonl` is absent. v3 metadata has `meta/tasks.parquet` with only `task_index: 0`. The first episode row in `meta/episodes/chunk-000/file-000.parquet` has `tasks = ["Pick up soda can and place it in the box"]`. | Prompt text exists through v3 episode metadata, not `tasks.jsonl`. |
| FPS, total frames, total episodes | `fps = 30`, `total_frames = 7402`, `total_episodes = 19`, `total_tasks = 1`. | Enough data for a short smoke run sanity check. |
| Gripper unit (`q01..q99` estimate) | Assuming contract gripper indices are `7` and `15`: state index 7 q01/q99 = `-0.7453 / -0.0342`; state index 15 q01/q99 = `-0.9300 / 0.0002`; action index 7 q01/q99 = `-0.7485 / -0.0344`; action index 15 q01/q99 = `-0.9299 / 0.0008`. | Not normalized `[0, 1]`; values look radian-like. This is a §3 unit/range risk for runtime compatibility, though not an H2 dimension mismatch. |
| Joint unit (degrees vs radians) | Global state min/max span roughly `-1.538..2.133`; global action min/max span roughly `-1.562..2.146`. First parquet q01/q99 ranges are also small radian-scale values, not degree-scale values. | Radians inferred. Spec says radians are a §3 risk for contract validation. |
| LeRobot `codebase_version` | `v3.0`; LeRobot metadata command loads successfully and reports `30 19 7402`. | v3 path applies; current loader compatibility layer is relevant for later gates. |

Conclusion: H1 on state/action dimensionality (16-D pos-only). Separate risks remain: semantic joint order is not named in metadata, and joint/gripper values appear radian-scale rather than contract degrees / normalized gripper.
