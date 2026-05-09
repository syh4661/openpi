# 작업 명세서 v3 — `pi05_openarm` (TetraxCode) 스모크 + 사용자 OpenArm 임바디먼트 어댑터

> **수신자**: Codex (학습 PC, 5090). 본 spec은 v1/v2를 대체한다.
> **선행 문서**: 본 레포 `CLAUDE.md`, `AGENTS.md`,
> `src/openpi/policies/openarm_runtime_contract.py`, `src/openpi/training/config.py`
> (origin/main 기준, 커밋 `e3c3961`까지 반영), 사이드 레포
> `../openarm_lerobot/docs/codex_handoff_pi05_plan.md` + `configs/record_full.json`.
>
> **v2와 결정적 차이**: v2는 origin/main 동기화 전 작성되어, T0 (중복 config)와 T1 일부 (TetraxCode
> repo_id 와이어업)가 이미 끝나 있다는 사실을 반영하지 못했다. v3은 그 현실 위에서 남은 작업만
> 남긴다.

---

## 0. 현재 origin/main 상태 (사실 확정, 재논의 금지)

다음은 origin/main의 코드를 직접 확인한 결과다. Codex는 이를 기반으로 시작한다.

### 0.1 이미 적용된 변경 (커밋 `e3c3961`, `27ee357`)

- **중복 `pi05_openarm` config 해결**: 기존 inference-only stub은 `pi05_openarm_runtime`으로
  rename됨. 풀 fine-tune config가 `pi05_openarm` 이름을 차지한다. 검증:
  ```
  $ uv run python -c "from openpi.training import config as c; print(len(c._CONFIGS))"
  → 33   (에러 없음)
  ```

- **`pi05_openarm` repo_id를 TetraxCode로 변경**: 학습 데이터 소스가
  `saurabh/openarm_pick_v6` → `TetraxCode/openarm_pick_place_v1`로 전환됨.

- **Camera repack 키 변경**:
  ```python
  "head":        "observation.images.cam_0"
  "wrist_left":  "observation.images.cam_2"
  "wrist_right": "observation.images.cam_4"
  ```
  TetraxCode 데이터셋의 카메라 컬럼명이 `cam_0/cam_2/cam_4`임을 의미. (cam_1/cam_3은 depth/IR로
  추정되나 정찰 §1.1에서 확인 필요.)

- **`asset_id`는 의도적으로 `saurabh/openarm_pick_v6` 유지**: norm stats를 saurabh 네임스페이스
  아래에 저장/로드하는 alias 구조. `compute_openarm_norm_stats.py`도 출력 경로를
  `assets/pi05_openarm/saurabh/openarm_pick_v6/`로 강제. 즉 학습 데이터는 TetraxCode에서 오지만
  norm stats 디렉토리는 saurabh 네임스페이스를 재사용한다.

- **`compute_openarm_norm_stats.py` fallback 로직 추가**:
  - `saurabh/openarm_pick_v6` 캐시가 있으면 그것을 사용
  - 없으면 `TetraxCode/openarm_pick_place_v1`을 자동 다운로드해서 stats 계산
  - 두 경로 모두 `chunk-*/episode_*.parquet`와 `chunk-*/file-*.parquet` glob 처리

- **`runtime_contract.py` config alias**: `pi05_openarm` 호출 시 `pi05_openarm_runtime`으로 자동
  라우팅. 기존 `openarm_inference.py` / `openarm_direct_runtime.py` / `openarm_policy_client.py`
  콜러는 이름 변경을 신경 쓸 필요가 없다.

- **`data_loader.py` LeRobot v2/v3 호환 레이어**: TetraxCode가 v3 포맷일 가능성을 흡수. Codex는
  로더 동작이 처음부터 안 풀리면 이 레이어를 먼저 의심.

### 0.2 아직 안 된 것 (본 spec의 작업 범위)

| 미확정 | 근거 |
|---|---|
| TetraxCode 데이터셋의 정확한 스키마 (state/action 차원, joint name 순서, gripper 단위, prompt 유무) | 코드는 16-D 가정으로 흘러가지만, 실제 차원이 16-D인지 미검증 |
| `assets/pi05_openarm/saurabh/openarm_pick_v6/norm_stats.json`의 존재 또는 신선도 | 로컬 캐시에 있는지 / TetraxCode 기반으로 계산되었는지 미확인 |
| `pi05_openarm` 학습이 실제로 끝까지 도는지 | 시도 기록 없음 |
| 사용자 OpenArm 로봇(`../openarm_lerobot`, 48-D right-first interleaved)과 openpi 컨트랙트(16-D left-first pos-only) 사이 어댑터 | v2 §4 그대로 살아있음 |
| 실로봇 no-send dry-run | 진행 안 됨 |

---

## 1. T1 — `pi05_openarm` (TetraxCode) 스모크 학습

### 1.1 데이터셋 정찰 (코드 한 줄 쓰기 전 필수)

산출물: `tmp/openarm_pickplace_v1_recon.md`. 다음을 표 형태로 채운다.

```bash
# meta만 받기 — 빠르고 디스크 절약
huggingface-cli download TetraxCode/openarm_pick_place_v1 --repo-type dataset \
    --include "meta/*" \
    --include "data/chunk-000/episode_000000.parquet"
```

확인 항목:

| 항목 | 채워야 할 값 | 분기 |
|---|---|---|
| `features.observation.state.shape` | ? | **16이 아니면 §3 risk** (현재 config는 16 가정) |
| `features.action.shape` | ? | 동상 |
| `state.names` (joint 순서) | ? | openpi 컨트랙트 STATE_ORDER (`left_*` 8 + `right_*` 8, .pos만)와 비교 |
| 카메라 컬럼명 정확히 (cam_0~cam_4 중 어떤 게 head/wrist_left/wrist_right인지) | ? | config의 `cam_0/cam_2/cam_4` 매핑이 맞는지 검증 |
| `tasks.jsonl` 존재 여부 + 첫 prompt | ? | `prompt_from_task=True` 라 prompt 없으면 학습 시 빈 문자열 또는 None 들어감 |
| FPS, total_frames, total_episodes | ? | 학습 step 수 sanity 체크 |
| Gripper 단위 (`q01..q99`로 추정) | ? | `[0,1]`이 아니면 §3 risk |
| Joint 단위 (degrees vs radians) | ? | radians면 §3 risk |
| LeRobot codebase_version (v2 / v3) | ? | v3면 `data_loader.py`의 v3 패치 경로 검증 |

이 표가 채워지기 전까지는 §1.2 이후로 진행 금지.

### 1.2 정규화 통계 계산

```bash
uv run scripts/compute_openarm_norm_stats.py
```

이 스크립트는 이미 fallback 로직이 들어가 있어서 saurabh 캐시 없으면 TetraxCode를 자동 다운로드한다.
산출물: `assets/pi05_openarm/saurabh/openarm_pick_v6/norm_stats.json`.

수용 기준:
- `state` / `actions` 둘 다 16차원으로 stats 계산되었음 (`mean.shape == (16,)`).
- 16개 차원 어느 것도 `std == 0`이거나 `q01 == q99`가 아님 (있으면 학습 발산 위험, 보고).
- 콘솔에 출력된 `Mean/Std/Q01/Q99`가 사람이 봤을 때 합리적 범위 (joint은 deg 가정 시 대략 ±100 이내,
  gripper는 [0,1] 가정 시 약 [0,1]).

만약 정찰 §1.1에서 state/action이 48-D라고 나오면, 본 스크립트는 48차원 stats를 만들 것이다 →
`OpenArmInputs`(16-D 가정)와 충돌. 그 경우 **§3 risk H2 분기**로 가서 dimension reduction transform을
추가해야 한다.

### 1.3 학습

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_openarm \
    --exp-name=tetrax_smoke \
    --overwrite
```

스모크 단계라 `num_train_steps`는 그대로 10k 두지 말고 **--num-train-steps=3000 (또는 더 짧게)**
오버라이드 권장. 풀 10k는 정합성 검증 후 별도 런으로.

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_openarm \
    --exp-name=tetrax_smoke \
    --num-train-steps=3000 \
    --save-interval=1000 \
    --keep-period=1000 \
    --overwrite
```

수용:
- 끝까지 OOM/normalization 에러 없이 진행.
- W&B/콘솔 loss 단조 감소 추세 (절대치 임계는 정찰 후 정함).
- 체크포인트 `checkpoints/pi05_openarm/tetrax_smoke/3000/` 생성, `assets/saurabh/openarm_pick_v6/`
  하위에 `norm_stats.json`이 동봉됨.

### 1.4 서빙 + 컨트랙트 검증

`pi05_openarm_runtime`(또는 alias `pi05_openarm`) 둘 다 같은 inference config로 라우팅됨.

```bash
# 직접 inference (체크포인트 + canonical fixture로 evidence 생성)
uv run scripts/openarm_direct_runtime.py \
    --config pi05_openarm_runtime \
    --checkpoint-dir ./checkpoints/pi05_openarm/tetrax_smoke/3000 \
    --fixture <path-to-canonical-fixture.npz> \
    --output /tmp/tetrax_smoke_evidence.npz \
    --default-prompt "pick the object and place it in the box"

# 별도 컨트랙트 검증 (fixture만으로 schema 검증)
uv run scripts/check_openarm_contract.py \
    --fixture <path-to-canonical-fixture.npz> \
    --contract-module openpi.policies.openarm_runtime_contract
```

수용:
- `evidence.npz` 안에 `(16, 16)` action chunk가 있고 `validate_action_chunk` 통과.
- joint 단위 degrees, gripper `[0, 1]` 범위 안.

### 1.5 (선택) 폴리시 서버

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_openarm_runtime \
    --policy.dir=./checkpoints/pi05_openarm/tetrax_smoke/3000
```

서버가 8000번에서 떠야 하고, 실제 클라이언트 호출은 §2 어댑터 작업 이후로 미룸.

---

## 2. T2 — 사용자 OpenArm ↔ openpi 임바디먼트 어댑터

이 절은 v2 §4와 본질이 같다. origin/main 변경에 영향받지 않는다. 요약만 다시 둠.

### 2.1 갭 매트릭스 (재확인)

| 차원 | 사용자 로봇 LeRobot 기록 | openpi 컨트랙트 | 흡수 |
|---|---|---|---|
| 차원 수 | 48 | 16 | inbound 어댑터에서 `.pos` 추출 |
| 모터 그룹 순서 | right-first | left-first | inbound 재정렬 |
| 필드 | (pos, vel, torque) interleaved | pos-only | inbound select |
| 카메라 키 | `chest`, `left_wrist`, `right_wrist` | `head`, `wrist_left`, `wrist_right` | `RUNTIME_CAMERA_NAME_TO_KEY` 이미 존재 |
| 이미지 layout | 480×640 | 224×224 CHW uint8 | inbound resize+transpose |
| Gripper 단위 | degrees `[-90, 45]` | `[0,1]` (0=open / 1=closed) | 양방향 변환 (실측 기반 상수) |

### 2.2 인덱스 상수

```python
# 양 모듈에서 import 해서 공유. 매직 넘버 금지.
LEROBOT_POS_INDICES_LEFT_FIRST = (
    24, 27, 30, 33, 36, 39, 42, 45,   # left_joint_1..7 + left_gripper .pos
     0,  3,  6,  9, 12, 15, 18, 21,   # right_joint_1..7 + right_gripper .pos
)
```
근거: `../openarm_lerobot/data/openarm_dualmanip_test04/meta/info.json`의 `observation.state.names`.

### 2.3 책임 분리

- **본 레포(openpi)는 16-D 컨트랙트만 유지**. 사용자 로봇 런타임 코드를 본 레포에 들이지 않는다.
- **48↔16 변환**은 `../openarm_lerobot/src/openarm_lerobot/openpi_bridge.py` 신설 모듈에 둔다.
  - `inbound(robot_obs_48 + 3 cams) -> openpi_obs_16 (CHW images, head/wrist_left/wrist_right)`
  - `outbound(openpi_action_16 chunk) -> robot_action_48 chunk`
  - gripper: degrees ↔ normalized `[0,1]` 변환 함수
- 단위 테스트는 사용자 측 레포에 둔다 (round-trip 4종 + gripper round-trip).

### 2.4 fixture

`../openarm_lerobot/data/<recent>/`에서 1 episode 1 frame 추출 → inbound 어댑터 적용 → canonical
fixture `.npz`로 저장. 이 fixture가 §1.4의 `--fixture` 인자.

---

## 3. Risks / 분기

| 발견 | 즉시 행동 |
|---|---|
| **H1**: TetraxCode `state`/`action`이 16-D pos-only | 그대로 §1.2 → §1.3 진행 |
| **H2**: TetraxCode가 48-D 또는 24-D 원본 | `OpenArmInputs`가 16-D 가정이라 학습 불가능. 다음 두 옵션 중 결정 보고: (a) `OpenArmInputs`에 dim reduction 추가, (b) 별도 `OpenArmLeRobotInputs` 신설. 본 spec은 (b)를 권장. 어느 쪽이든 `pi05_openarm`을 in-place로 깨지 말고 새 config 추가 권장 |
| Joint이 radian | 학습은 가능, 컨트랙트 검증에서 fail. 정찰 보고 → 사람 결정 |
| Gripper sign이 사용자 로봇과 반대 | 학습은 가능, outbound 어댑터에서 mirror |
| 카메라가 cam_0/2/4 가 아닌 다른 인덱스 | config repack RHS 수정 (in-place OK, 단일 commit) |
| LeRobot v3 포맷이고 `data_loader.py` v3 패치가 아직 미흡 | 로그를 먼저 보고, 수정 범위가 넓으면 사람과 결정 |
| `safe_bi_openarm_follower.send_action`이 vel/torque를 의미 있게 사용 | 0.0 패딩 위험. outbound 어댑터에서 처리 결정 |

---

## 4. Codex 사용 가이드

- **AGENTS.md / CLAUDE.md / 본 spec을 먼저 읽고 시작**.
- **승인 모드는 `Code Review` (또는 동급 review-only)**. `Auto` 금지 — H1/H2 분기 결정 같은 구조적
  선택을 자동 머지하면 회귀 위험.
- **Atomic 커밋**:
  - 정찰 결과 (`tmp/openarm_pickplace_v1_recon.md`만 추가) — 1 커밋
  - norm stats 계산 결과물은 별도 PR/커밋 필요 없음 (assets는 .gitignore되거나 별도 push 정책 따름)
  - (필요 시) config / openarm_policy.py 수정 — H2 분기 시 1~2 커밋
  - 학습 로그 / evidence는 git에 커밋하지 말 것 (`tmp/` 또는 W&B로 빠짐)
- **금지**:
  - `openarm_runtime_contract.py` 수정 (이번 작업 범위 아님; 이미 `e3c3961`에서 alias 추가됨)
  - `pi05_openarm` 또는 `pi05_openarm_runtime` config 이름 변경
  - 실로봇 send-action (T3 게이트 통과 후 별도 spec)
  - `git add -A` (private/세션 파일 섞임)
- **검증 명령**: spec에서 직접 인용. 발명 금지.
- **컨벤션**: Python 3.11, ruff line-length 120, `*_test.py` 모듈 옆에 위치, `@pytest.mark.manual`로
  GPU/네트워크/체크포인트 의존 테스트 게이트.

---

## 5. Claude 측 review/검증 스킬 chain (Codex 산출 후, 사람이 invoke)

| 단계 | 스킬 | 목적 |
|---|---|---|
| 정찰 후 | (없음, 사람이 표 검토) | H1/H2 분기 결정 |
| H2 분기 코드 추가 후 | `/python-review` | 인덱스 상수 / 매직 넘버 / type hint |
| 어댑터 단위 테스트 추가 후 | `/tdd` | round-trip 4종 + gripper round-trip 강제 |
| 어댑터 본체 후 (T2 끝) | `/santa-loop` (선택, 무겁다) | 실로봇 위험 직결 코드 → 이중 리뷰 합리적 |
| T3 직전 | `/security-review` | 폴리시 서버 노출, websocket 권한, fixture 경로 leakage |
| 마무리 | `/prp-pr` | PR 본문 자동 생성 + 산출물 체크리스트 |
| (보조) 한 PR 단위 검토 | `/code-review` | 변경 diff 일반 검토 |

---

## 6. 산출물 체크리스트

**openpi 레포**
- [ ] `tmp/openarm_pickplace_v1_recon.md` (정찰 결과 표 채움)
- [ ] (H2 분기 시) `src/openpi/policies/openarm_lerobot_adapter.py` 신설 + 그에 맞는 새 config
- [ ] `assets/pi05_openarm/saurabh/openarm_pick_v6/norm_stats.json` 생성 (TetraxCode 기반)
- [ ] `checkpoints/pi05_openarm/tetrax_smoke/3000/` 생성 (또는 운영자가 정한 step)
- [ ] `openarm_direct_runtime.py` evidence `.npz` 생성, `validate_action_chunk` PASS

**openarm_lerobot 레포 (T2)**
- [ ] `src/openarm_lerobot/openpi_bridge.py` 신설 (inbound/outbound + gripper 변환)
- [ ] 단위 테스트 (round-trip 4종 + gripper round-trip)
- [ ] canonical fixture 생성 스크립트 또는 일회용 노트북

**T3 (별도 spec으로 분기 가능)**
- [ ] no-send dry-run 로그
- [ ] operator 명시적 `GO`

---

## 7. 비범위

- LoRA / FSDP / 멀티노드.
- PyTorch 학습 경로 (`train_pytorch.py`). 첫 스모크는 JAX.
- `openarm_runtime_contract.py` 수정 (alias 이상의 추가 변경 금지).
- 실로봇 send-action.
- Hugging Face dataset/checkpoint push.
- 자체 데이터(leader-follower 또는 Quest로 수집한 사용자 데이터) 학습 + 라이브 데모 — 별도 spec.
- v2에서 제안했던 `pi05_openarm_pickplace_v1` 신규 config 추가 — origin/main이 `pi05_openarm`을
  TetraxCode로 직접 와이어업했으므로 더 이상 필요 없음. **추가 시 회귀로 간주**.
