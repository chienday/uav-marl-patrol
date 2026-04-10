# UAV Patrol — Multi-Agent Reinforcement Learning

> **Đồ án tốt nghiệp** — Lê Đức Chiến (2251172253) · 64KTPM3 · Trường Đại học Thủy Lợi  

Nghiên cứu và xây dựng mô hình tuần tra khép kín đội UAV trong đô thị bằng học tăng cường đa tác tử (MARL).

---

## Mục lục

- [Mô tả bài toán](#mô-tả-bài-toán)
- [Môi trường mô phỏng](#môi-trường-mô-phỏng)
- [Các thuật toán triển khai](#các-thuật-toán-triển-khai)
- [Kết quả sơ bộ](#kết-quả-sơ-bộ)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt & chạy](#cài-đặt--chạy)
- [Dataset & Maps](#dataset--maps)
- [Model đã huấn luyện](#model-đã-huấn-luyện)

---

## Mô tả bài toán

Bài toán tuần tra đô thị bằng đội UAV được mô hình hóa dưới dạng **Multi-Agent Reinforcement Learning (MARL)**:

- Mỗi UAV là một **agent** hoạt động độc lập nhưng cùng mục tiêu chung
- Mục tiêu: **tối đa hóa coverage** (diện tích bao phủ) trong thời gian ngắn nhất
- Ràng buộc: tránh chướng ngại vật, tránh va chạm giữa UAV, không bay vào vùng cấm

**Tại sao MARL?**  
Môi trường đô thị có không gian trạng thái lớn, nhiều UAV cần phối hợp mà không cần giao tiếp trực tiếp. MARL (đặc biệt MAPPO, VDPPO) cho phép học chiến lược phân tán hiệu quả hơn lập lịch thủ công.

---

## Môi trường mô phỏng

### Địa hình (Grid Maps)

Môi trường được xây dựng dưới dạng **lưới 2D 10×10**, mỗi ô là một khu vực đô thị. Có 3 kịch bản bản đồ:

| Bản đồ | Kích thước | Obstacles | Ô tự do | Max steps | Đặc điểm |
|--------|-----------|-----------|---------|-----------|-----------|
| **Simple Map** | 10×10 | 4 ô | 96 | 800 | Obstacles rải rác, dễ navigate |
| **Mixed Map** | 10×10 | 7 ô | 93 | 800 | Tạo nhiều hành lang phức tạp |
| **Bottleneck Map** | 10×10 | 9 ô | 91 | 1000 | Hàng rào row 5, chỉ 1 ô thông tại [5,4] |


### State Space (305 chiều — IPPO 2 UAV)

| Thành phần | Chiều | Mô tả |
|-----------|-------|-------|
| Coverage map | 100 | Ô nào đã được thăm (0/1) |
| Obstacle map | 100 | Vị trí chướng ngại vật cố định |
| Visit count (chuẩn hóa) | 100 | Số lần UAV thăm mỗi ô |
| Vị trí UAV hiện tại | 2 | Tọa độ (x,y) chuẩn hóa [0,1] |
| BFS distance to frontier | 1 | Khoảng cách đến ô chưa thăm gần nhất |
| Vị trí UAV còn lại | 2 | Tọa độ UAV khác (chỉ IPPO/MARL) |

### Action Space

4 hành động rời rạc: `UP(0)`, `DOWN(1)`, `LEFT(2)`, `RIGHT(3)`

### Reward Function (10 thành phần)

| Thành phần | Công thức | Giá trị |
|-----------|-----------|---------|
| Khám phá ô mới | +EXPLORE_REWARD | +60 |
| Va chạm obstacle | −OBSTACLE_PENALTY | −20 |
| Revisit penalty | −min(visits² × 1.5, 25) | −1.5 đến −25 |
| Passage bonus | +PASSAGE_BONUS (1 lần) | +80 |
| Frontier reward | +frontier_count × 12 | 0 đến +48 |
| BFS guidance | +8 / (dist+1) | +0.4 đến +8 |
| Coverage bonus | +coverage_ratio × 25 | 0 đến +25 |
| Step penalty | −0.1/bước | −0.1 |
| Complete bonus (≥97%) | +3000 | +3000 |
| Partial bonus (truncated) | +coverage_ratio × 600 | 0 đến +600 |
| **Collision penalty** *(MARL only)* | −15 (khi 2 UAV cùng ô) | −15 |

---

## Các thuật toán triển khai

### 1. PPO đơn tác tử (Baseline)

File: `uav-rl-final.ipynb`

- Thuật toán: **Proximal Policy Optimization** (Stable-Baselines3)
- Kiến trúc mạng: MLP [512, 256, 128]
- 8 parallel envs (DummyVecEnv + VecNormalize)
- 10M timesteps training

### 2. IPPO — Independent PPO 2 UAV *(đang phát triển)*


- Mỗi UAV chạy PPO độc lập, **dùng chung 1 policy** (parameter sharing)
- Môi trường: `UAVMultiAgentEnv` — 2 UAV cùng lúc
- Wrapper: `IPPOWrapper` — biến multi-agent thành Gym đơn cho SB3
- Observation: 305 chiều (thêm vị trí UAV còn lại)
- Collision penalty khi 2 UAV vào cùng ô

**Kiến trúc IPPO:**

```
        ┌─────────────────────────────────┐
        │         Shared Policy           │
        │        MLP [512→256→128]        │
        └────────────┬────────────────────┘
                     │  (parameter sharing)
           ┌─────────┴─────────┐
      obs_UAV0             obs_UAV1
      (305-dim)            (305-dim)
           │                   │
        action_0           action_1
           │                   │
        UAVMultiAgentEnv (step cả 2 cùng lúc)
```

### 3. MAPPO & VDPPO *(đang phát triển)*

- MAPPO: Centralized Critic + Decentralized Actor
- VDPPO: Value Decomposition + Mixing Network (credit assignment)

---

## Kết quả sơ bộ

### PPO đơn tác tử (10M steps — đã hoàn thành)

| Bản đồ | Coverage TB | Std | ≥90% | Bước đến 90% |
|--------|------------|-----|------|-------------|
| Simple Map | **94.6%** ✓ | 7.7 | 17/20 | ~116 |
| Mixed Map | **97.8%** ✓ | 2.1 | 19/20 | ~129 |
| Bottleneck Map | **97.8%** ✓ | 3.4 | 18/20 | ~117 |

Cải thiện so với Random Agent: **+5.3%** (Simple), **+15.7%** (Mixed), **+19.5%** (Bottleneck)

### IPPO 2 UAV *(đang phát triển)*
### MAPPO & VDPPO *(đang phát triển)*


## Cài đặt & chạy

### Yêu cầu

```bash
Python >= 3.10
stable-baselines3[extra] >= 2.0
gymnasium >= 0.29
torch >= 2.0
numpy, matplotlib
```

### Cài đặt

```bash
git clone https://github.com/chienday/uav-marl-patrol.git
pip install stable-baselines3[extra] gymnasium matplotlib numpy
```

### Chạy PPO đơn tác tử
Link kaggle thực tế : https://www.kaggle.com/code/chienday/uav-rl-final
```bash
# Trên Kaggle (có GPU T4/P100):
# Mở notebooks/uav-rl-final.ipynb → Run All

# Local (CPU):
jupyter notebook notebooks/uav-rl-final.ipynb
```

### Chạy IPPO 2 UAV *(đang phát triển)*
### Chạy MAPPO và VDPPO *(đang phát triển)*




## Dataset & Maps

Bản đồ được định nghĩa dưới dạng JSON, dễ chỉnh sửa:

```json
{
  "name": "bottleneck",
  "grid_size": 10,
  "obstacles": [
    [5,0],[5,1],[5,2],[5,3],
    [5,5],[5,6],[5,7],[5,8],[5,9]
  ],
  "start_position": [0, 0],
  "max_steps": 1000
}
```

Để tạo bản đồ mới, thêm file JSON vào thư mục `maps/` và truyền đường dẫn vào `UAVPatrolEnv(map_file=...)`.

---

## Model đã huấn luyện

| Model | Thuật toán | Steps | Coverage TB | Download |
|-------|-----------|-------|------------|---------|
| `ppo_uav_v3.zip` | PPO đơn | 10M | 96.7% | [Kaggle Output] |
| `best_model.zip` | PPO đơn (best) | ~7M | ~97% | [Kaggle Output] |


---

## Liên hệ

**Lê Đức Chiến** · leducchien2002@gmail.com · 0378910817  
Trường Đại học Thủy Lợi — Khoa Công nghệ Thông tin
