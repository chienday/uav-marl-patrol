# UAV Patrol — Multi-Agent Reinforcement Learning

> **Đồ án tốt nghiệp** — Lê Đức Chiến (2251172253) · 64KTPM3 · Trường Đại học Thuỷ Lợi

Nghiên cứu và xây dựng mô hình tuần tra khép kín đội UAV trong đô thị bằng học tăng cường đa tác tử (MARL).

---

## Mục lục

- [Mô tả bài toán](#mô-tả-bài-toán)
- [Môi trường mô phỏng](#môi-trường-mô-phỏng)
- [Các thuật toán triển khai](#các-thuật-toán-triển-khai)
- [Kết quả thực nghiệm](#kết-quả-thực-nghiệm)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt và chạy](#cài-đặt-và-chạy)
- [Dataset và Maps](#dataset-và-maps)
- [Model đã huấn luyện](#model-đã-huấn-luyện)

---

## Mô tả bài toán

Bài toán tuần tra đô thị bằng đội UAV được mô hình hoá dưới dạng **Multi-Agent Reinforcement Learning (MARL)**:

- Mỗi UAV là một **agent** hoạt động độc lập nhưng cùng mục tiêu chung
- Mục tiêu: **tối đa hoá coverage** (diện tích bao phủ) trong thời gian ngắn nhất
- Ràng buộc: tránh chướng ngại vật, tránh va chạm giữa các UAV, không bay vào vùng cấm

**Tại sao MARL?**
Môi trường đô thị có không gian trạng thái lớn, nhiều UAV cần phối hợp mà không cần giao tiếp trực tiếp. MARL (đặc biệt MAPPO, VDPPO) cho phép học chiến lược phân tán hiệu quả hơn lập lịch thủ công.

---

## Môi trường mô phỏng

### Địa hình (Grid Maps)

Môi trường được xây dựng dưới dạng **lưới 2D 10×10**, mỗi ô là một khu vực đô thị. Có 3 kịch bản bản đồ:

| Bản đồ | Kích thước | Chướng ngại vật | Ô tự do | Số bước tối đa | Đặc điểm |
|--------|:---------:|:---------------:|:-------:|:--------------:|-----------|
| **Simple** | 10×10 | 7 ô | 93 | 600 | Chướng ngại vật rải rác (hàng 2, 5, 7) |
| **Mixed** | 10×10 | 16 ô | 84 | 600 | Nhiều tường ngang tạo hành lang |
| **Bottleneck** | 10×10 | 16 ô | 84 | 800 | Hàng rào hàng 4, chỉ 1 ô thông tại [4,4] |

### Không gian quan sát (Observation Space)

Mỗi agent sử dụng **cửa sổ cục bộ** (local window) thay vì toàn bộ bản đồ:

| Thuật toán | OBS_RADIUS | Cửa sổ | Số chiều obs | Mô tả |
|-----------|:----------:|:------:|:------------:|-------|
| **IPPO** | 2 | 5×5 | **80** | 25×3 kênh + 5 đặc trưng |
| **MAPPO** | 4 | 9×9 | **248** | 81×3 kênh + 5 đặc trưng |
| **VDPPO** | 4 | 9×9 | **248** | 81×3 kênh + 5 đặc trưng |

Chi tiết vector quan sát:

| Thành phần | Kích thước | Mô tả |
|-----------|:----------:|-------|
| Cửa sổ coverage | L | Ô nào đã được thăm (0/1) trong cửa sổ |
| Cửa sổ chướng ngại vật | L | Vật cản (pad=1 ngoài biên) |
| Số lần thăm (chuẩn hoá) | L | Số lần UAV đã thăm, chuẩn hoá [0,1] |
| Vị trí UAV hiện tại | 2 | Toạ độ (x/G, y/G) chuẩn hoá |
| Vị trí tương đối UAV khác | 2 | (dx/2G, dy/2G) cắt [-1,1] |
| Khoảng cách BFS đến biên giới | 1 | Khoảng cách BFS đến ô chưa thăm / (2G) |

*(L = LOCAL_SIZE = (2×OBS_RADIUS+1)²)*

### Trạng thái toàn cục (MAPPO/VDPPO — CTDE)

Critic mạng trung tâm sử dụng **global state** (304 chiều) gồm: bản đồ coverage + bản đồ chướng ngại vật + số lần thăm (toàn bộ 10×10) + vị trí 2 UAV.

### Không gian hành động

4 hành động rời rạc: `LÊN(0)`, `XUỐNG(1)`, `TRÁI(2)`, `PHẢI(3)`

Hai UAV hành động **đồng thời** (simultaneous), không có thiên lệch thứ tự.

### Hàm thưởng (Reward Function)

Ba cấu hình thưởng riêng biệt cho 3 thuật toán:

| Thành phần | IPPO | MAPPO | VDPPO |
|-----------|:----:|:-----:|:-----:|
| Thưởng khám phá ô mới | 60.0 | 5.0 | 5.0 |
| Phạt va chạm chướng ngại vật | 20.0 | 2.0 | 2.0 |
| Thang phủ sóng | 25.0 | 3.0 | 3.0 |
| Thang biên giới | 12.0 | 1.5 | 1.5 |
| Hệ số thăm lại | 1.5 | 0.5 | 1.5 |
| Giới hạn phạt thăm lại | 25.0 | 5.0 | 35.0 |
| Thang BFS | 8.0 | 1.0 | 1.0 |
| Phạt mỗi bước | 0.1 | 0.05 | 1.5 |
| Thưởng hoàn thành (≥97%) | 300.0 | 100.0 | 100.0 |
| Thang hoàn thành một phần | 60.0 | 20.0 | 20.0 |
| Thưởng qua lối hẹp | 8.0 | 3.0 | 3.0 |
| Phạt va chạm giữa UAV | 15.0 | 3.0 | 25.0 |
| Phạt chồng lấn | 5.0 | 1.0 | 12.0 |
| Hệ số thưởng đội | 0.3 | 2.0 | 2.5 |

> **Ghi chú:** IPPO dùng thang thưởng lớn vì SB3 PPO có VecNormalize tự động chuẩn hoá returns. MAPPO/VDPPO dùng custom trainer với Welford online return normalization nên thang thưởng nhỏ hơn.

### Phát hiện va chạm (3 loại)

1. **Va chạm trực diện (Head-on)**: 2 UAV di chuyển vào cùng 1 ô → cả 2 bị trả về vị trí cũ
2. **Va chạm chéo (Cross)**: 2 UAV đổi chỗ vị trí (swap) → cả 2 bị trả về vị trí cũ
3. **Chồng lấn (Overlap)**: 2 UAV ở cùng 1 ô sau khi di chuyển → phạt OVERLAP_PENALTY

---

## Các thuật toán triển khai

### 1. PPO đơn tác tử (Baseline)

- **File**: `notebooks/ppo-final.ipynb`
- Thuật toán: **Proximal Policy Optimization** (Stable-Baselines3)
- Kiến trúc mạng: MLP [512, 256, 128]
- 8 môi trường song song (DummyVecEnv + VecNormalize)
- 10 triệu bước huấn luyện

### 2. IPPO — Independent PPO (2 UAV)

- **File**: `notebooks/ippo-final.ipynb` | `src/agents/ippo_trainer.py`
- Mỗi UAV chạy **SB3 PPO độc lập**, luân phiên huấn luyện (alternating rounds)
- Wrapper: `SingleAgentWrapper` — biến multi-agent thành Gym đơn cho SB3
- 3 vòng × 2 triệu bước/vòng, quan sát 80 chiều (cửa sổ 5×5)

**Siêu tham số IPPO:**

| Tham số | Giá trị |
|---------|---------|
| Tốc độ học | 2e-4 |
| Số bước thu thập | 2048 |
| Kích thước batch | 256 |
| Số epoch | 10 |
| Gamma | 0.995 |
| GAE Lambda | 0.95 |
| Phạm vi cắt | 0.2 |
| Hệ số entropy | 0.04 |

```
  Vòng 1: Huấn luyện Agent-0 (đối tác=ngẫu nhiên) → Huấn luyện Agent-1 (đối tác=Agent-0)
  Vòng 2: Huấn luyện Agent-0 (đối tác=Agent-1) → Huấn luyện Agent-1 (đối tác=Agent-0)
  Vòng 3: Huấn luyện Agent-0 (đối tác=Agent-1) → Huấn luyện Agent-1 (đối tác=Agent-0)
```

### 3. MAPPO — Multi-Agent PPO (CTDE)

- **File**: `notebooks/mappo-final.ipynb` | `src/agents/mappo_trainer.py`
- Kiến trúc: **Huấn luyện tập trung, Thực thi phân tán (CTDE)**
- **SharedActor**: dùng chung cho 2 agent, kích hoạt Tanh, khởi tạo trực giao
- **CentralCritic**: nhận trạng thái toàn cục (304 chiều), Tanh, bộ tối ưu riêng
- Vòng lặp huấn luyện PyTorch tuỳ chỉnh, GAE theo agent, cắt hàm mất mát giá trị
- Chuẩn hoá lợi nhuận trực tuyến Welford

```
          SharedActor (obs 248 → logits 4)
               |  (Tanh, khởi tạo trực giao)
          +---------+---------+
        obs_0 (248)       obs_1 (248)
          |                   |
       action_0           action_1
          |                   |
   UAVPatrolEnvIPPO (bước đồng thời)
          |
    CentralCritic (global_state 304 → V_team)
```

### 4. VDPPO — Value-Decomposed PPO

- **File**: `notebooks/vdppo-final.ipynb` | `src/agents/vdppo_trainer.py`
- Mở rộng MAPPO với **Phân rã giá trị (Value Decomposition)**:
  - `V_tổng = V_đội + trung_bình(V_agent_i)`
  - `V_đội`: CentralCritic (Tanh) — giá trị đội
  - `V_agent_i`: AgentValueHead (ReLU) — giá trị cá nhân
- **Entropy thích ứng**: tự động tăng/giảm hệ số entropy theo entropy chính sách
- **Thưởng đa dạng**: phạt khi 2 UAV quá gần (khoảng cách Manhattan ≤ 2)

```
  V_tổng = CentralCritic(trạng_thái_toàn_cục)         — V_đội
         + trung_bình( AgentValueHead_i(obs_i) )        — V_cá_nhân
```

---

## Kết quả thực nghiệm

### PPO đơn tác tử (10 triệu bước)

| Bản đồ | Coverage TB | Độ lệch | Tỉ lệ ≥90% | Bước đến 90% |
|--------|:----------:|:-------:|:----------:|:------------:|
| Simple | **94.6%** | 7.7 | 17/20 | ~116 |
| Mixed | **97.8%** | 2.1 | 19/20 | ~129 |
| Bottleneck | **97.8%** | 3.4 | 18/20 | ~117 |

Cải thiện so với Agent ngẫu nhiên: **+5.3%** (Simple), **+15.7%** (Mixed), **+19.5%** (Bottleneck)

### IPPO 2 UAV (3 vòng × 2 triệu bước)

| Bản đồ huấn luyện | Simple | Mixed | Bottleneck | Trung bình |
|-------------------|:------:|:-----:|:----------:|:----------:|
| Simple | **75.8%** | 62.4% | 74.5% | 70.9% |
| Mixed | 75.1% | **65.6%** | 67.4% | 69.4% |
| Bottleneck | 68.5% | 55.7% | **67.5%** | 63.9% |

Tiến triển qua các vòng (coverage phối hợp):

| Vòng | Simple | Mixed | Bottleneck | Trung bình |
|:----:|:------:|:-----:|:----------:|:----------:|
| 1 | 59.7% | 49.3% | 62.8% | 57.3% |
| 2 | 55.5% | 51.5% | 66.2% | 57.7% |
| 3 | **67.3%** | **63.2%** | **77.8%** | **69.4%** |

### MAPPO (197 lần cập nhật, 8 môi trường)

| Bản đồ | Coverage | Chồng lấn | Entropy |
|--------|:--------:|:---------:|:-------:|
| Simple | **95.7%** | 5.4 | 0.71 |
| Mixed | **68.1%** | 0.2 | 0.77 |
| Bottleneck | **97.9%** | 2.8 | 0.11 |
| **Trung bình** | **87.2%** | 2.8 | 0.53 |

### VDPPO (87 lần cập nhật, 3.5 triệu bước)

| Bản đồ | Coverage | Chồng lấn | Entropy |
|--------|:--------:|:---------:|:-------:|
| Simple | **97.8%** | 2.0 | 0.35 |
| Mixed | **97.9%** | 1.6 | 0.18 |
| Bottleneck | **97.6%** | 0.2 | 0.46 |
| **Trung bình** | **97.8%** | 1.3 | 0.33 |

### So sánh tổng hợp

| Thuật toán | Simple | Mixed | Bottleneck | Trung bình | Chồng lấn |
|-----------|:------:|:-----:|:----------:|:----------:|:---------:|
| PPO (1 UAV) | 94.6% | 97.8% | 97.8% | 96.7% | — |
| IPPO (2 UAV) | 75.8% | 65.6% | 67.5% | 69.6% | Cao |
| MAPPO (2 UAV) | 95.7% | 68.1% | 97.9% | 87.2% | 2.8 |
| **VDPPO (2 UAV)** | **97.8%** | **97.9%** | **97.6%** | **97.8%** | **1.3** |

> **Nhận xét:** VDPPO đạt hiệu suất cao nhất trên cả 3 bản đồ với chồng lấn thấp nhất, cho thấy phân rã giá trị + entropy thích ứng giúp 2 UAV phối hợp hiệu quả.

---

## Cấu trúc thư mục

```
uav_marl_patrol_final/
│
├── notebooks/                      # Sổ tay Jupyter (Kaggle)
│   ├── ppo-final.ipynb             # PPO đơn tác tử (baseline)
│   ├── ippo-final.ipynb            # IPPO — 2 agent độc lập
│   ├── mappo-final.ipynb           # MAPPO — CTDE
│   └── vdppo-final.ipynb           # VDPPO — Phân rã giá trị
│
├── src/                            # Mã nguồn (module hoá, có thể import)
│   ├── envs/
│   │   ├── uav_env_single.py       # UAVPatrolEnv (PPO baseline)
│   │   ├── uav_env_multi.py        # UAVPatrolEnvIPPO (IPPO/MAPPO/VDPPO)
│   │   └── reward.py               # RewardConfig + IPPORewardConfig + VDPPORewardConfig
│   │
│   ├── agents/
│   │   ├── ippo_trainer.py         # IPPOTrainer + SingleAgentWrapper
│   │   ├── mappo_trainer.py        # MAPPOTrainer (PyTorch tuỳ chỉnh)
│   │   ├── vdppo_trainer.py        # VDPPOTrainer (mở rộng MAPPO)
│   │   └── networks.py             # SharedActor, CentralCritic, AgentValueHead
│   │
│   └── utils/
│       └── visualization.py        # Tiện ích vẽ biểu đồ
│
├── maps/                           # Bản đồ (JSON)
│   ├── map_simple.json
│   ├── map_mixed.json
│   └── map_bottleneck.json
│
├── checkpoints/                    # Trọng số mô hình
│   ├── ppo/                        # ppo_uav_v3.zip
│   ├── ippo/                       # agent0_final.zip, agent1_final.zip
│   ├── mappo/                      # mappo_uav_final_actor.pt, _critic.pt
│   └── vdppo/                      # vdppo_uav_optimized_actor.pt, _critic.pt, _agent_heads.pt
│
├── log/                            # Kết quả thực nghiệm (từ Kaggle)
│   ├── ppo_single_uav/             # Biểu đồ + mô hình
│   ├── ippo_experiment/            # Cấu hình, đánh giá, biểu đồ
│   ├── mappo_experiment/           # Lịch sử đánh giá 197 bản ghi
│   └── vdppo_experiment/           # Lịch sử đánh giá 87 bản ghi
│
├── plots/                          # Biểu đồ tách riêng từng thuật toán
│   ├── ppo/
│   ├── ippo/
│   ├── mappo/
│   ├── vdppo/
│   └── comparison/
│
├── scripts/
│   └── generate_plots.py           # Script tạo biểu đồ từ log
│
├── requirements.txt
└── .gitignore
```

---

## Cài đặt và chạy

### Yêu cầu

```
Python >= 3.10
torch >= 2.0
gymnasium >= 0.29
stable-baselines3[extra] >= 2.0
numpy, matplotlib
```

### Cài đặt

```bash
git clone https://github.com/chienday/uav-marl-patrol.git
cd uav-marl-patrol
pip install -r requirements.txt
```

### Chạy trên Kaggle

| Thuật toán | Liên kết Kaggle |
|-----------|----------------|
| PPO | https://www.kaggle.com/code/chienday/uav-rl-final |
| IPPO | https://www.kaggle.com/code/chienday/ippo-final |
| MAPPO | https://www.kaggle.com/code/chienday/mappo-final |
| VDPPO | https://www.kaggle.com/code/chienday/vdppo-final |

### Chạy cục bộ (sử dụng module src)

```python
from src.envs import UAVPatrolEnvIPPO, RewardConfig, VDPPORewardConfig
from src.agents import MAPPOTrainer, VDPPOTrainer

# Huấn luyện MAPPO
trainer = MAPPOTrainer(
    map_file="maps/map_simple.json",
    map_paths={
        "simple":     "maps/map_simple.json",
        "mixed":      "maps/map_mixed.json",
        "bottleneck": "maps/map_bottleneck.json",
    },
    total_steps=4_000_000,
)
trainer.train()
trainer.save()

# Huấn luyện VDPPO
vdppo = VDPPOTrainer(
    map_file="maps/map_simple.json",
    map_paths={...},
    total_steps=3_500_000,
)
vdppo.train()
vdppo.save()
```

### Tạo biểu đồ

```bash
python scripts/generate_plots.py
# Kết quả: plots/ppo/, plots/ippo/, plots/mappo/, plots/vdppo/, plots/comparison/
```

---

## Dataset và Maps

Bản đồ được định nghĩa dưới dạng JSON, dễ chỉnh sửa:

```json
{
  "name": "bottleneck",
  "grid_size": 10,
  "max_steps": 800,
  "start_position": [0, 0],
  "obstacles": [
    [4,0],[4,1],[4,2],[4,3],
    [4,5],[4,6],[4,7],[4,8],[4,9],
    [2,2],[2,3],[6,6],[6,7],
    [8,1],[8,2],[8,3]
  ]
}
```

Để tạo bản đồ mới, thêm file JSON vào thư mục `maps/` và truyền đường dẫn vào `UAVPatrolEnvIPPO(map_file=...)`.

---

## Model đã huấn luyện

| Model | Thuật toán | Số bước | Coverage |
|-------|-----------|:-------:|:--------:|
| `checkpoints/ppo/ppo_uav_v3.zip` | PPO đơn | 10 triệu | 96.7% |
| `checkpoints/ippo/agent{0,1}_final.zip` | IPPO | 3×2 triệu | 69.4% |
| `checkpoints/mappo/mappo_uav_final_{actor,critic}.pt` | MAPPO | ~800 nghìn | 87.2% |
| `checkpoints/vdppo/vdppo_uav_optimized_{actor,critic,agent_heads}.pt` | VDPPO | 3.5 triệu | **97.8%** |

---

## Liên hệ

**Lê Đức Chiến** · leducchien2002@gmail.com · 0378910817
Trường Đại học Thuỷ Lợi — Khoa Công nghệ Thông tin
