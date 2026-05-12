# GUIDE TOÀN DIỆN — UAV MARL Patrol
## Dành cho người học AI/ML, chưa chuyên sâu RL

> **Lê Đức Chiến (2251172253)** — Đại học Thuỷ Lợi — 2026  
> File này bao gồm: Lý thuyết từ zero · Cài đặt · Đọc code · Mở rộng · Bảo vệ đồ án · Bộ câu hỏi Hội đồng

---

## MỤC LỤC

- [Phần 1 — Lý thuyết từ zero đến VDPPO](#phần-1--lý-thuyết-từ-zero-đến-vdppo)
- [Phần 2 — Cài đặt môi trường](#phần-2--cài-đặt-môi-trường)
- [Phần 3 — Chạy lại thực nghiệm](#phần-3--chạy-lại-thực-nghiệm)
- [Phần 4 — Đọc hiểu mã nguồn](#phần-4--đọc-hiểu-mã-nguồn)
- [Phần 5 — Mở rộng và tuỳ chỉnh](#phần-5--mở-rộng-và-tuỳ-chỉnh)
- [Phần 6 — Viết báo cáo và trình bày](#phần-6--viết-báo-cáo-và-trình-bày)
- [Phần 7 — Bộ câu hỏi Hội đồng](#phần-7--bộ-câu-hỏi-hội-đồng)

---

# PHẦN 1 — LÝ THUYẾT TỪ ZERO ĐẾN VDPPO

## 1.1 Reinforcement Learning là gì? — Khác gì ML thông thường?

### So sánh ba loại học máy

```
┌──────────────────┬──────────────────────┬────────────────────────┐
│ Supervised (Học  │ Unsupervised (Học     │ Reinforcement (Học     │
│ có giám sát)     │ không giám sát)      │ Tăng cường)            │
├──────────────────┼──────────────────────┼────────────────────────┤
│ Có nhãn đúng/sai │ Không có nhãn        │ Không có nhãn đúng     │
│ cho mỗi mẫu      │                      │ → tự khám phá          │
│                  │                      │                        │
│ Ví dụ: nhận dạng │ Ví dụ: phân cụm      │ Ví dụ: học chơi game,  │
│ ảnh, dịch thuật  │ khách hàng           │ điều khiển robot       │
│                  │                      │                        │
│ Teacher nói:     │ Tự tìm cấu trúc      │ Nhận điểm thưởng/phạt  │
│ "Đây là con mèo" │ trong dữ liệu        │ → học từ trial & error │
└──────────────────┴──────────────────────┴────────────────────────┘
```

### Ví dụ trực quan — UAV học tuần tra

Hãy tưởng tượng bạn đang dạy một chú UAV bay qua khu phố:

- **Supervised Learning:** Ghi lại đường bay của phi công chuyên nghiệp → UAV bắt chước. Vấn đề: tốn công ghi lại, và chỉ làm được những gì đã thấy.
- **RL:** UAV tự thử → nếu bay vào tường bị phạt điểm → nếu khám phá vùng mới được thưởng → dần dần học được chiến lược tốt nhất. Có thể vượt qua con người!

### Vòng lặp RL cơ bản

```
        ┌─────────────────────────────────────────────────┐
        │                                                 │
        │  Bước t:                                        │
        │  1. Agent nhìn trạng thái sₜ (bản đồ + vị trí) │
        │  2. Agent chọn hành động aₜ (đi Lên/Xuống/...)  │
        │  3. Môi trường trả về:                          │
        │     - Trạng thái mới sₜ₊₁                       │
        │     - Phần thưởng rₜ (+điểm nếu khám phá mới,  │
        │                        -điểm nếu đâm tường)    │
        │  4. Lặp lại đến khi kết thúc episode           │
        │                                                 │
        │  Mục tiêu: Tối đa hoá TỔNG phần thưởng:        │
        │  G = r₀ + γ·r₁ + γ²·r₂ + ...                  │
        │      (γ = 0.99 = "tương lai gần bằng hiện tại") │
        └─────────────────────────────────────────────────┘
```

> **Lưu ý quan trọng:** Hệ số chiết khấu γ (gamma = 0.995 trong VDPPO) gần bằng 1 có nghĩa là agent rất coi trọng phần thưởng xa trong tương lai — phù hợp với bài toán coverage dài hơi.

---

## 1.2 Markov Decision Process (MDP) — Ngôn ngữ toán học của RL

MDP là cách viết chặt chẽ bài toán RL. Đừng sợ — nó chỉ là định nghĩa chính xác 5 thứ:

| Ký hiệu | Tên | Trong bài toán UAV |
|---------|-----|--------------------|
| **S** | Tập trạng thái | Tất cả vị trí có thể của 2 UAV + bản đồ đã thăm |
| **A** | Tập hành động | {Lên, Xuống, Trái, Phải} |
| **P(s'|s,a)** | Xác suất chuyển | Nếu đi Lên từ [3,4] → đến [2,4] (hoặc [3,4] nếu có tường) |
| **R(s,a)** | Phần thưởng | +5 khám phá ô mới, -2 đâm tường, +100 hoàn thành |
| **γ** | Hệ số chiết khấu | 0.995 trong VDPPO |

**Tính chất Markov** (quan trọng!): Quyết định tiếp theo chỉ cần biết trạng thái **hiện tại**, không cần nhớ toàn bộ lịch sử. Điều này giúp bài toán tractable.

---

## 1.3 Policy và Value Function — Hai khái niệm cốt lõi

### Policy (Chính sách) π

Là hàm quyết định của agent: **"Ở trạng thái s, làm hành động gì?"**

```
Deterministic policy:  π(s) = a         (luôn làm hành động cụ thể)
Stochastic policy:     π(a|s) = P(a|s)  (phân phối xác suất trên các hành động)

Ví dụ (stochastic):
  Ở ô [3,4], cửa sổ quan sát cho thấy bên phải có ô chưa thăm:
  π(Phải | [3,4]) = 0.7  ← 70% xác suất đi phải (có ô mới)
  π(Lên   | [3,4]) = 0.2  ← 20% xác suất đi lên
  π(Xuống | [3,4]) = 0.05
  π(Trái  | [3,4]) = 0.05
```

### Value Function (Hàm giá trị)

Là ước tính **"từ trạng thái s này, tôi kỳ vọng sẽ kiếm được bao nhiêu điểm tổng cộng?"**

```
V(s) = Kỳ vọng tổng phần thưởng tương lai nếu bắt đầu từ s

Ví dụ:
  V([0,0]) = 350  ← Từ góc xuất phát, kỳ vọng kiếm 350 điểm
  V([4,4]) = 280  ← Từ lối hẹp Bottleneck, kỳ vọng thấp hơn (khó khám phá)
  V([9,9]) = 200  ← Góc xa, đã thăm nhiều rồi, ít điểm còn lại
```

### Advantage Function A(s,a)

**"Hành động a này tốt hơn hay tệ hơn so với trung bình ở trạng thái s?"**

```
A(s,a) = Q(s,a) - V(s)
       = (điểm nếu làm a) - (điểm kỳ vọng trung bình)

A(s, Phải) = +5  → Đi phải tốt hơn kỳ vọng (+5 điểm)
A(s, Trái) = -3  → Đi trái tệ hơn kỳ vọng  (-3 điểm)
```

Advantage được dùng để cập nhật policy: **tăng xác suất hành động có A > 0, giảm A < 0**.

---

## 1.4 GAE — Generalized Advantage Estimation

Trong thực tế, ta không biết A(s,a) chính xác → phải ước tính. GAE là cách ước tính cân bằng:

```
Bước 1: Tính TD error tại mỗi bước t:
  δₜ = rₜ + γ·V(sₜ₊₁) - V(sₜ)
       ↑         ↑          ↑
  phần thưởng  giá trị   giá trị
  nhận được   bước sau   bước này

Bước 2: Tính GAE:
  Âₜ = δₜ + (γλ)·δₜ₊₁ + (γλ)²·δₜ₊₂ + ...

  λ = 0 → Chỉ dùng bước hiện tại (ít variance, nhiều bias)
  λ = 1 → Dùng toàn bộ episode (ít bias, nhiều variance)
  λ = 0.95-0.97 → Cân bằng tốt (dùng trong đề tài)
```

> **Tại sao cần GAE?** Nếu chỉ dùng δₜ (1 bước) → ước tính thiếu chính xác. Nếu dùng cả episode (Monte Carlo) → phương sai quá lớn. GAE cân bằng giữa hai cực.

---

## 1.5 PPO — Proximal Policy Optimization

PPO là thuật toán giải quyết vấn đề: **"Cập nhật policy bao nhiêu là đủ, không quá ít cũng không quá nhiều?"**

### Vấn đề với gradient ascent thông thường

```
Cập nhật thông thường: θ ← θ + α·∇J(θ)

Vấn đề:
  Nếu α quá lớn → policy thay đổi đột ngột → sụp đổ (catastrophic forgetting)
  Nếu α quá nhỏ → học chậm, lãng phí dữ liệu

Ví dụ thực tế:
  Episode 1: UAV học đi phải rất hiệu quả
  Update: tăng xác suất "đi phải" lên 95% (quá nhiều!)
  Episode 2: UAV chỉ biết đi phải → kẹt ở tường bên phải
  → Policy sụp đổ!
```

### Giải pháp PPO — Clipped Objective

```python
# Tỷ lệ xác suất mới / cũ
ratio = π_new(a|s) / π_old(a|s)

# Hàm mục tiêu clipped:
L = min(
    ratio * Advantage,
    clip(ratio, 1-0.2, 1+0.2) * Advantage
)

Ý nghĩa trực quan:
  Nếu Advantage > 0 (hành động tốt):
    → Tăng xác suất, nhưng KHÔNG quá 1.2 lần so với cũ
  Nếu Advantage < 0 (hành động xấu):
    → Giảm xác suất, nhưng KHÔNG dưới 0.8 lần so với cũ
```

### Vòng lặp PPO

```
repeat:
  1. Thu thập T=2048 bước với policy hiện tại
  2. Tính GAE Advantage cho mỗi bước
  3. Cập nhật policy K=10 epoch:
     for mini-batch in data:
         tính L_PPO + L_value + entropy_bonus
         gradient.step()
  4. Loại bỏ dữ liệu cũ (on-policy!)
```

> **Điểm khác với DQN/Q-Learning:** PPO là **on-policy** — chỉ học từ dữ liệu mình vừa thu thập, không có replay buffer. Điều này ổn định hơn nhưng kém sample-efficient hơn DQN.

---

## 1.6 MARL — Tại sao cần khi có nhiều Agent?

### Vấn đề khi áp dụng RL đơn vào đa tác tử

```
Thử dùng 2 PPO độc lập (→ đây chính là IPPO):

  Agent-0 học: "Đi về góc phải dưới là tốt"
  Agent-1 cũng học: "Đi về góc phải dưới là tốt"

  Kết quả: Cả 2 đi về góc phải dưới → chồng lấn → coverage thấp!

Vấn đề cốt lõi:
  ✗ Môi trường không dừng: Agent-1 đang học, policy Agent-0 thay đổi
    → Môi trường của Agent-1 thay đổi → training không ổn định
  ✗ Không có cơ chế phân chia lãnh thổ
  ✗ Khó phân biệt: "Reward cao vì tôi tốt hay vì agent kia tốt?"
```

### Ba paradigm chính của MARL

```
Paradigm 1: FULLY DECENTRALIZED (IPPO)
──────────────────────────────────────
Agent-0 học từ obs_0, không biết gì về Agent-1
Agent-1 học từ obs_1, không biết gì về Agent-0
  ✓ Đơn giản, mở rộng được
  ✗ Non-stationary, khó phối hợp

Paradigm 2: FULLY CENTRALIZED
──────────────────────────────
1 agent siêu to nhận toàn bộ quan sát, đưa ra toàn bộ quyết định
  ✓ Có thể phối hợp hoàn hảo
  ✗ Không mở rộng được (hành động tăng theo hàm mũ)
  ✗ Không triển khai thực tế (cần liên lạc liên tục)

Paradigm 3: CTDE — Centralized Training, Decentralized Execution ★
──────────────────────────────────────────────────────────────────
KHI HUẤN LUYỆN: Critic truy cập thông tin toàn cục
KHI TRIỂN KHAI: Actor chỉ dùng quan sát cục bộ
  ✓ Phối hợp tốt khi train
  ✓ Triển khai thực tế được (không cần giao tiếp khi bay)
  → Đây là paradigm của MAPPO và VDPPO!
```

---

## 1.7 MAPPO — Multi-Agent PPO (CTDE)

```
KIẾN TRÚC MAPPO:

LÚC HUẤN LUYỆN:
                  obs_0 (248 chiều)
                       │
              ┌────────┴────────┐
              │  SharedActor    │  ← Dùng chung 1 mạng cho cả 2 agent
              │ 248→256→128→4   │    (parameter sharing)
              └────────┬────────┘
                  action_0 (phân phối xác suất)
                  
  global_state ──► CentralCritic ──► V_team
  (304 chiều)     (304→256→128→1)    (giá trị đội)
  [coverage_map + obstacle_map + visit_count + pos_0 + pos_1]

LÚC THỰC THI:
  obs_0 ──► SharedActor ──► action_0   (không cần CentralCritic!)
  obs_1 ──► SharedActor ──► action_1
```

### Tại sao dùng SharedActor (chia sẻ tham số)?

```
Nếu Agent-0 và Agent-1 dùng actor riêng:
  → Mỗi agent học từ ~50% dữ liệu
  → Học chậm hơn

Nếu dùng SharedActor:
  → Cả 2 agent đều học từ 100% dữ liệu
  → Học nhanh gấp đôi
  → Hợp lý vì 2 UAV có cùng nhiệm vụ (symmetric)
```

---

## 1.8 VDPPO — Value Decomposition PPO (Đỉnh của đề tài)

### Vấn đề MAPPO chưa giải quyết được

```
Tình huống:
  Agent-0: vừa khám phá 5 ô mới → reward cao
  Agent-1: vừa đâm vào tường   → reward âm
  
  Team reward = (reward_0 + reward_1) / 2 → trung bình thấp

  CentralCritic(global_state) → V_team = 150 (trung bình)

  Kết quả: Gradient cập nhật Actor không phân biệt được
  → Agent-0 không được khuyến khích đúng mức
  → Agent-1 không bị phạt đúng mức
  → Vấn đề Credit Assignment!
```

### Giải pháp VDPPO — Phân rã giá trị

```python
# Thay vì chỉ có V_team:
V_total(s) = V_team(global_state)            # CentralCritic (Tanh)
           + mean(V_agent_i(obs_i))          # AgentValueHead (ReLU)

# Ví dụ cụ thể:
V_team    = 200  (giá trị đội chung từ global state)
V_agent_0 = 80   (đóng góp cá nhân Agent-0 — khám phá tốt)
V_agent_1 = 20   (đóng góp cá nhân Agent-1 — bị kẹt)
V_total   = 200 + (80+20)/2 = 250

→ Gradient cho Actor của Agent-0 phản ánh đóng góp của Agent-0!
→ Gradient cho Actor của Agent-1 phản ánh đóng góp của Agent-1!
→ Credit Assignment chính xác hơn!
```

### Entropy Thích ứng

```
Entropy H[π] = -Σ π(a|s)·log π(a|s)
  = 0    → policy hoàn toàn deterministic (luôn chọn 1 hành động)
  = ln4  → policy hoàn toàn random (4 hành động bằng nhau)

Vấn đề với entropy cố định:
  ent_coef quá cao → agent random mãi, không học được chiến lược
  ent_coef quá thấp → agent bị local optimum sớm

VDPPO giải quyết bằng adaptive entropy:
  if entropy_hiện_tại < entropy_mục_tiêu:
      ent_coef *= 1.1    # Tăng khám phá
  else:
      ent_coef *= 0.995  # Giảm dần → hội tụ
```

### Diversity Reward — Ép 2 UAV chia nhau

```python
# Nếu 2 UAV quá gần nhau:
distance = |x_0 - x_1| + |y_0 - y_1|  # Manhattan distance
if distance <= 2:
    penalty = OVERLAP_PENALTY × (3 - distance)
    reward_0 -= penalty
    reward_1 -= penalty

# Hiệu ứng:
# 2 UAV giữ khoảng cách ≥ 3 ô
# → Tự nhiên chia đôi bản đồ để khám phá
# → Coverage tăng từ 68% (MAPPO) lên 97.9% (VDPPO) trên Mixed map!
```

---

# PHẦN 2 — CÀI ĐẶT MÔI TRƯỜNG

## 2.1 Yêu cầu hệ thống

| Thành phần | Tối thiểu | Khuyên dùng |
|-----------|:---------:|:-----------:|
| Python | 3.10+ | 3.11 |
| RAM | 8 GB | 16 GB |
| Ổ cứng | 2 GB | 5 GB |
| GPU | Không cần | CUDA 11.8+ (train nhanh hơn 3-5×) |
| OS | Windows 10 / Ubuntu 20.04 | Ubuntu 22.04 |

## 2.2 Cài đặt local

```bash
# Bước 1: Clone repo
git clone https://github.com/chienday/uav-marl-patrol.git
cd uav-marl-patrol

# Bước 2: Tạo virtual environment (khuyên dùng)
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows

# Bước 3: Cài thư viện
pip install torch>=2.0
pip install gymnasium>=0.29
pip install stable-baselines3[extra]>=2.0
pip install numpy matplotlib scipy

# Bước 4: Kiểm tra
python -c "import torch; import gymnasium; import stable_baselines3; print('OK!')"
```

## 2.3 Chạy trên Kaggle (khuyên dùng để train)

Kaggle cung cấp GPU/CPU miễn phí, đây là nơi đề tài đã chạy thực nghiệm:

```
1. Vào https://www.kaggle.com → Đăng nhập
2. Chọn "Code" → "New Notebook"
3. Upload notebooks/vdppo-final.ipynb
4. Chọn "Session options" → GPU T4 x2 (hoặc CPU nếu chỉ test)
5. Nhấn "Run All"
6. Tải kết quả về từ /kaggle/working/
```

---

# PHẦN 3 — CHẠY LẠI THỰC NGHIỆM

## 3.1 Cấu trúc thực nghiệm

```
Mỗi thuật toán có 1 notebook chạy trên Kaggle:

notebooks/ppo-final.ipynb    → 10M bước, ~2-3 giờ (CPU Kaggle)
notebooks/ippo-final.ipynb   → 3×2M bước, ~4-5 giờ
notebooks/mappo-final.ipynb  → 4M bước, ~5-6 giờ
notebooks/vdppo-final.ipynb  → 3.5M bước, ~4-5 giờ

Kết quả được lưu vào /kaggle/working/logs/ và /checkpoints/
→ Download về sau khi chạy xong
```

## 3.2 Reproduce kết quả VDPPO (nhanh nhất)

```python
# Option 1: Load checkpoint đã có (không cần train lại)
import torch
from src.agents.networks import SharedActor

actor = SharedActor(obs_dim=248, action_dim=4)
actor.load_state_dict(torch.load('checkpoints/vdppo/vdppo_uav_optimized_actor.pt'))
actor.eval()

# Option 2: Train lại từ đầu (500K bước để test nhanh)
from src.agents import VDPPOTrainer

trainer = VDPPOTrainer(
    map_file="maps/map_simple.json",
    map_paths={
        "simple":     "maps/map_simple.json",
        "mixed":      "maps/map_mixed.json",
        "bottleneck": "maps/map_bottleneck.json",
    },
    total_steps=500_000,     # 3.5M trong thực nghiệm đầy đủ
    n_envs=4,                # 8 trong thực nghiệm đầy đủ
    seed=42,
)
trainer.train()
trainer.save("my_vdppo")
```

## 3.3 Phân tích kết quả từ log

```python
import json
import matplotlib.pyplot as plt

# Đọc lịch sử VDPPO
with open('logs/vdppo_experiment/eval_history.json') as f:
    history = json.load(f)

# Vẽ đường cong hội tụ
updates = [h['update'] for h in history]
coverage_mean = [h['eval']['mean']['coverage'] for h in history]

plt.plot(updates, coverage_mean, label='VDPPO Mean Coverage')
plt.axhline(y=97.8, color='red', linestyle='--', label='Final 97.8%')
plt.xlabel('Update')
plt.ylabel('Coverage (%)')
plt.title('VDPPO Convergence Curve')
plt.legend()
plt.savefig('vdppo_convergence.png', dpi=150)
```

## 3.4 So sánh các thuật toán từ log hiện có

```python
# Dữ liệu có sẵn trong logs/
logs = {
    'MAPPO':  'logs/mappo_experiment/eval_history.json',
    'VDPPO':  'logs/vdppo_experiment/eval_history.json',
}

for name, path in logs.items():
    with open(path) as f:
        data = json.load(f)
    final = data[-1]['eval']['mean']['coverage']
    print(f"{name}: Final coverage = {final:.1f}%")
```

---

# PHẦN 4 — ĐỌC HIỂU MÃ NGUỒN

## 4.1 Luồng dữ liệu tổng quan

```
Bước 1: Khởi tạo
  maps/map_simple.json ──► UAVPatrolEnvIPPO(map_file=...)
                           → tạo grid 10×10, đặt 2 UAV tại [0,0]

Bước 2: Thu thập dữ liệu (Rollout)
  env.reset() → obs_0, obs_1 (mỗi cái 248 chiều)
  SharedActor(obs_0) → action_0 (0/1/2/3)
  SharedActor(obs_1) → action_1
  env.step(action_0, action_1) → obs_new, rewards, done, info
  [lặp 512 bước × 8 envs = 4096 dữ liệu mỗi update]

Bước 3: Tính giá trị
  CentralCritic(global_state) → V_team
  AgentValueHead_0(obs_0) → V_agent_0
  AgentValueHead_1(obs_1) → V_agent_1
  V_total = V_team + (V_agent_0 + V_agent_1) / 2

Bước 4: Tính Advantage (GAE)
  δₜ = rₜ + γ·V_total(sₜ₊₁) - V_total(sₜ)
  Âₜ = Σ (γλ)^k · δₜ₊ₖ

Bước 5: Cập nhật mạng (8 epoch)
  L_actor = PPO Clip(ratio, Â)
  L_critic = MSE(V_pred, V_target)
  L_total = L_actor - ent_coef·Entropy + vf_coef·L_critic
  optimizer.step()
```

## 4.2 File networks.py — Trái tim của hệ thống

```python
# ============= SharedActor =============
class SharedActor(nn.Module):
    def __init__(self, obs_dim=248, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),   # ← Tanh (không phải ReLU!)
            nn.Linear(256, 256),     nn.Tanh(),
            nn.Linear(256, 128),     nn.Tanh(),
            nn.Linear(128, action_dim)
        )
        # Khởi tạo trực giao (orthogonal init) → ổn định hơn random
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)

    def forward(self, obs):
        logits = self.net(obs)
        return Categorical(logits=logits)  # Phân phối xác suất trên 4 hành động

# ============= CentralCritic =============
class CentralCritic(nn.Module):
    def __init__(self, global_state_dim=304):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(304, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 1)    # → V_team (scalar)
        )

# ============= AgentValueHead =============
class AgentValueHead(nn.Module):
    def __init__(self, obs_dim=248):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(248, 128), nn.ReLU(),  # ← ReLU (khác với Actor dùng Tanh)
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 1)     # → V_agent_i (scalar)
        )
```

> **Tại sao Actor dùng Tanh, ValueHead dùng ReLU?**
> - Actor cần output logits có thể âm hoặc dương → Tanh cho hidden layers ổn định hơn
> - ValueHead ước tính giá trị cá nhân (thường dương trong bài toán hợp tác) → ReLU đủ dùng và nhanh hơn

## 4.3 File uav_env_multi.py — Môi trường

```python
class UAVPatrolEnvIPPO(gym.Env):

    def reset(self):
        # Đặt 2 UAV về vị trí start_position
        # Reset coverage map (tất cả = 0)
        # Reset visit count
        return [obs_agent0, obs_agent1]

    def step(self, actions):
        # actions = [action_0, action_1]  (0/1/2/3)

        # 1. Xử lý va chạm TRƯỚC KHI di chuyển
        new_pos = [move(pos_0, actions[0]), move(pos_1, actions[1])]
        if head_on_collision(pos_0, pos_1, new_pos):
            new_pos = [pos_0, pos_1]  # Trả về vị trí cũ
            collision = True

        # 2. Tính BFS distance từ mỗi UAV đến ô chưa thăm gần nhất
        bfs_dist_0 = bfs_frontier(new_pos[0], coverage_map)
        bfs_dist_1 = bfs_frontier(new_pos[1], coverage_map)

        # 3. Tính reward cho từng agent
        reward_0 = compute_reward(agent=0, new_pos=new_pos, ...)
        reward_1 = compute_reward(agent=1, new_pos=new_pos, ...)

        # 4. Tính observation mới (cửa sổ 9×9 × 3 kênh + 5 đặc trưng)
        obs_0 = get_observation(agent=0, new_pos=new_pos, ...)
        obs_1 = get_observation(agent=1, new_pos=new_pos, ...)

        return [obs_0, obs_1], [reward_0, reward_1], done, info

    def get_global_state(self):
        # Dùng cho CentralCritic (chỉ khi huấn luyện)
        # 100 (coverage) + 100 (obstacles) + 100 (visits) + 2 + 2 = 304
        return np.concatenate([
            coverage_map.flatten(),      # 100 chiều
            obstacle_map.flatten(),      # 100 chiều
            visit_count.flatten() / max, # 100 chiều (chuẩn hoá)
            pos_0 / grid_size,           # 2 chiều
            pos_1 / grid_size,           # 2 chiều
        ])
```

## 4.4 File reward.py — Hàm thưởng

```python
# Hàm thưởng VDPPO — 14 thành phần
def compute_reward(agent_id, new_pos, old_pos, coverage_map, ...):

    r = 0.0

    # 1. Thưởng khám phá ô mới
    if coverage_map[new_pos] == 0:
        r += EXPLORE_REWARD          # +5.0

    # 2. Thưởng theo % coverage hiện tại
    coverage_pct = coverage_map.sum() / n_free_cells
    r += COVERAGE_SCALE * coverage_pct  # +3.0 × 0.97 = +2.91

    # 3. Thưởng tiến về ô biên giới (frontier)
    old_bfs = bfs_to_frontier(old_pos)
    new_bfs = bfs_to_frontier(new_pos)
    if new_bfs < old_bfs:
        r += FRONTIER_SCALE          # +1.5

    # 4. Thưởng BFS (tiến gần ô chưa thăm)
    r += BFS_SCALE * (old_bfs - new_bfs) / (2*GRID)   # +1.0

    # 5. Phạt thời gian (mỗi bước)
    r -= STEP_PENALTY                # -1.5 (VDPPO)

    # 6. Phạt va chạm chướng ngại vật
    if obstacle_map[new_pos]:
        r -= OBSTACLE_PENALTY        # -2.0

    # 7. Phạt va chạm giữa 2 UAV
    if collision:
        r -= COLLISION_PENALTY       # -25.0 (VDPPO rất cao!)

    # 8. Phạt chồng lấn (2 UAV cùng ô)
    if overlap:
        r -= OVERLAP_PENALTY         # -12.0

    # 9. Thưởng hoàn thành (coverage ≥ 97%)
    if coverage_pct >= 0.97:
        r += COMPLETE_BONUS          # +100.0

    # 10. Thưởng team (coverage chung tăng)
    r += TEAM_ALPHA * team_coverage_gain  # +2.5 × gain

    return r
```

---

# PHẦN 5 — MỞ RỘNG VÀ TUỲ CHỈNH

## 5.1 Tạo bản đồ mới

```json
// Tạo file maps/map_my_city.json
{
  "name": "my_city",
  "grid_size": 10,
  "max_steps": 700,
  "start_position": [0, 0],
  "obstacles": [
    [1, 0], [1, 1], [1, 2],
    [3, 5], [3, 6], [3, 7], [3, 8],
    [6, 2], [6, 3], [6, 4],
    [8, 6], [8, 7]
  ]
}
```

```python
# Dùng bản đồ mới
trainer = VDPPOTrainer(
    map_file="maps/map_my_city.json",
    map_paths={
        "simple":  "maps/map_simple.json",
        "my_city": "maps/map_my_city.json",  # Thêm vào evaluation
    }
)
```

## 5.2 Điều chỉnh hàm thưởng

```python
# Tạo config thưởng tuỳ chỉnh
from src.envs.reward import VDPPORewardConfig
from dataclasses import dataclass

@dataclass
class MyRewardConfig(VDPPORewardConfig):
    EXPLORE_REWARD: float  = 8.0    # Tăng thưởng khám phá (mặc định: 5.0)
    COLLISION_PENALTY: float = 30.0  # Phạt va chạm nặng hơn
    STEP_PENALTY: float = 0.5        # Ít phạt thời gian hơn

env = UAVPatrolEnvIPPO(
    map_file="maps/map_simple.json",
    reward_config=MyRewardConfig()
)
```

## 5.3 Thêm thuật toán mới (ví dụ: QMIX-PPO)

```python
# Cấu trúc cần implement:
class QMIXPPOTrainer:
    def __init__(self, ...):
        self.actor_0 = IndividualActor(obs_dim=248)  # Actor riêng
        self.actor_1 = IndividualActor(obs_dim=248)  # Actor riêng
        self.mixer   = QMIXNetwork(n_agents=2, state_dim=304)
        # QMIX: Q_total = mixer(Q_0, Q_1, global_state)
        # Ràng buộc: ∂Q_total/∂Q_i >= 0 (monotonicity)

    def compute_value(self, obs_0, obs_1, global_state):
        q_0 = self.actor_0.get_value(obs_0)
        q_1 = self.actor_1.get_value(obs_1)
        q_total = self.mixer(q_0, q_1, global_state)
        return q_total
```

## 5.4 Mở rộng lên 3 UAV

```python
# Thay đổi trong uav_env_multi.py:
N_AGENTS = 3  # Thay vì 2

# Trong vdppo_trainer.py:
# Vòng lặp rollout cần lặp qua N_AGENTS agent
# Global state cần thêm vị trí UAV thứ 3 (2 chiều)
# → global_state từ 304 → 306 chiều

# Trong networks.py:
# V_total = V_team + (V_0 + V_1 + V_2) / 3
```

---

# PHẦN 6 — VIẾT BÁO CÁO VÀ TRÌNH BÀY

## 6.1 Cấu trúc báo cáo đồ án

```
Bố cục chuẩn (~80-100 trang):

Trang bìa + Lời cam đoan + Mục lục + Danh sách hình/bảng
│
├── Chương 1: Giới thiệu (5-8 trang)
│   ├── Bối cảnh và động lực
│   ├── Mục tiêu đề tài
│   ├── Phạm vi giới hạn
│   ├── Nghiên cứu liên quan  ← QUAN TRỌNG — HĐ hay hỏi
│   └── Đóng góp chính
│
├── Chương 2: Cơ sở lý thuyết (15-20 trang)
│   ├── RL + MDP + Policy Gradient
│   ├── PPO
│   ├── MARL (Dec-POMDP, CTDE)
│   ├── IPPO, MAPPO, VDPPO
│   └── Value Decomposition
│
├── Chương 3: Thiết kế môi trường (10-15 trang)
│   ├── 3 loại bản đồ (với hình ảnh!)
│   ├── Observation Space (248 chiều)
│   ├── Action Space
│   ├── Reward Function (14 thành phần)
│   └── Collision Detection (3 loại)
│
├── Chương 4: Kiến trúc hệ thống (15-20 trang)
│   ├── Sơ đồ kiến trúc 4 thuật toán (hình vẽ!)
│   ├── Chi tiết mạng neural
│   └── Vòng lặp huấn luyện
│
├── Chương 5: Kết quả thực nghiệm (15-20 trang)
│   ├── Thiết lập thực nghiệm
│   ├── Kết quả từng thuật toán (có bảng + hình!)
│   ├── So sánh tổng hợp
│   └── Phân tích thống kê (mean ± std)
│
├── Chương 6: Thảo luận (8-12 trang)
│   ├── Coordination Gap IPPO
│   ├── Tại sao MAPPO thất bại trên Mixed
│   ├── Tại sao VDPPO thành công
│   └── So sánh với nghiên cứu liên quan
│
└── Chương 7: Kết luận (3-5 trang)
    ├── Tóm tắt đóng góp
    ├── Hạn chế
    └── Hướng phát triển

Tài liệu tham khảo (12+ nguồn)
Phụ lục (config JSON, raw data)
```

## 6.2 Mẹo viết báo cáo hiệu quả

### Với bảng kết quả
```
✓ Luôn có cột "Trung bình" cho toàn bộ bản đồ
✓ In đậm giá trị tốt nhất trong mỗi cột
✓ Thêm cột Overlap và Entropy — không chỉ Coverage
✓ Thêm mean ± std để chứng minh độ tin cậy
✗ Đừng chỉ báo cáo "best result" — HĐ sẽ hỏi về variance
```

### Với hình ảnh
```
✓ Dùng ảnh PNG từ logs/*/plots/ — đây là kết quả THỰC
✓ Mỗi hình cần có caption giải thích ngắn gọn
✓ Hình kiến trúc mạng phải có kích thước các lớp
✗ Đừng dùng hình từ internet thay cho kết quả thực của mình
```

### Với so sánh
```
✓ So sánh công bằng: cùng map, cùng seed, cùng metric
✓ Giải thích tại sao số liệu này tốt/xấu — không chỉ báo cáo
✓ Thừa nhận điểm yếu của phương pháp mình
✗ Đừng so sánh với paper dùng môi trường khác mà không nói rõ
```

## 6.3 Slide thuyết trình (15-20 phút)

```
Gợi ý phân bổ thời gian cho slide (15 phút):

Slide 1-2:   Đề tài là gì? Tại sao quan trọng? (2 phút)
Slide 3-4:   Bài toán cụ thể + 3 bản đồ (2 phút)
Slide 5-7:   4 thuật toán (sơ đồ kiến trúc) (3 phút)
Slide 8-10:  Kết quả chính + Bảng so sánh (3 phút)
Slide 11-12: Phân tích Coordination Gap + VDPPO wins (2 phút)
Slide 13:    Kết luận + Hướng phát triển (1 phút)
Slide 14:    Q&A (dành thời gian cho HĐ)

Nguyên tắc slide:
  ✓ Tối đa 5-7 dòng/slide
  ✓ Hình ảnh chiếm 50%+ diện tích slide
  ✓ Font >= 24pt
  ✓ Số liệu kết quả phải xuất hiện trên slide
```

---

# PHẦN 7 — BỘ CÂU HỎI HỘI ĐỒNG

> Được tổng hợp từ kinh nghiệm thực tế. Mỗi câu có: **Câu hỏi → Điểm chính cần trả lời → Mức độ khó**

---

## 7.1 Câu hỏi về Lý thuyết RL/MARL

---

**Q1.** PPO khác gì so với DQN? Tại sao đề tài chọn PPO mà không dùng DQN?
> ★★☆ (Trung bình)

**Điểm chính:**
- DQN: value-based, off-policy, học Q(s,a), chỉ dùng được với discrete action → phù hợp khi space nhỏ
- PPO: policy-based, on-policy, học trực tiếp π(a|s) → ổn định hơn, mở rộng sang continuous được
- Lý do chọn PPO: on-policy ổn định hơn trong môi trường đa tác tử; SB3 có sẵn VecNormalize tiện lợi; MAPPO (paper nền) dùng PPO

---

**Q2.** Tại sao dùng GAE thay vì Monte Carlo returns để tính Advantage?
> ★★★ (Khó)

**Điểm chính:**
- Monte Carlo: dùng toàn bộ episode → **ít bias, nhiều variance** (vì episode dài, nhiều yếu tố ngẫu nhiên)
- TD(0): chỉ 1 bước → **ít variance, nhiều bias** (chỉ nhìn 1 bước tới, không đủ)
- GAE với λ=0.95: cân bằng → **bias và variance vừa phải** — kết quả tốt hơn cả hai
- Công thức: Âₜ = Σₖ (γλ)^k · δₜ₊ₖ — cộng có trọng số giảm dần

---

**Q3.** CTDE là gì? Tại sao không dùng fully decentralized hoặc fully centralized?
> ★★☆

**Điểm chính:**
- Fully decentralized (IPPO): đơn giản nhưng non-stationary, khó phối hợp (kết quả 69.6%)
- Fully centralized: cần liên lạc liên tục → không triển khai thực tế được
- CTDE: lúc train Critic dùng global state (phối hợp tốt), lúc deploy Actor chỉ dùng local obs (thực tế được)
- Trong đề tài: CentralCritic thấy 304 chiều nhưng Actor chỉ thấy 248 chiều

---

**Q4.** Value Decomposition trong VDPPO hoạt động như thế nào? Khác gì MAPPO?
> ★★★

**Điểm chính:**
- MAPPO: V_total = V_team(global_state) — 1 giá trị chung cho đội
- VDPPO: V_total = V_team + mean(V_agent_i) — phân rã thành team + cá nhân
- V_agent_i học từ obs_i → gán credit đúng cho từng agent
- Ví dụ: Agent-0 khám phá tốt → V_agent_0 cao → gradient Actor-0 được thưởng đúng
- Kết quả: Mixed map 68.1% (MAPPO) → 97.9% (VDPPO) — cải thiện +29.8%

---

**Q5.** Tại sao IPPO Joint Coverage (69%) thấp hơn coverage cá nhân (91%)? Giải thích bằng lý thuyết.
> ★★★

**Điểm chính:**
- Joint coverage = % ô được ít nhất 1 agent thăm / tổng ô tự do
- Nếu 2 agent đi **cùng vùng** → joint coverage chỉ bằng 1 agent → ~91%
- Coordination Gap = 91% - 69% = 22% = % ô bị cả 2 cùng thăm thay vì phân chia
- Nguyên nhân: không có cơ chế chia lãnh thổ; cùng xuất phát [0,0]; non-stationary
- Minh chứng: qua 3 vòng, gap giảm dần (57%→69%) nhưng không đủ thời gian để hội tụ

---

**Q6.** Entropy trong RL là gì? Tại sao cần thêm entropy vào hàm mục tiêu PPO?
> ★★☆

**Điểm chính:**
- Entropy H[π] = -Σ π(a|s)·log π(a|s) — đo mức độ ngẫu nhiên của policy
- Cao (=ln4≈1.386): policy hoàn toàn random → khám phá nhiều
- Thấp (→0): policy deterministic → khai thác (exploitation)
- Nếu không có entropy: agent converge quá sớm vào local optimum
- Ví dụ: sẽ học đi theo hành lang cố định thay vì khám phá toàn bản đồ
- Adaptive entropy của VDPPO: tự điều chỉnh để cân bằng exploration/exploitation

---

**Q7.** Dec-POMDP là gì và môi trường trong đề tài có phải là Dec-POMDP không?
> ★★★ (Nâng cao)

**Điểm chính:**
- Dec-POMDP: mô hình toán học cho bài toán đa tác tử với quan sát cục bộ (partially observable)
- Trong đề tài: Có — mỗi UAV chỉ thấy cửa sổ 9×9 quanh mình, không thấy toàn bản đồ → POMDP
- Global state 304 chiều chỉ dùng cho Critic khi train, không dùng khi deploy → đúng Dec-POMDP spirit
- Thách thức: Dec-POMDP tổng quát là NP-hard/NEXP-complete → cần approximate solution (→ VDPPO)

---

## 7.2 Câu hỏi về Thiết kế hệ thống

---

**Q8.** Tại sao observation space có kênh BFS distance? Có thể bỏ đi được không?
> ★★☆

**Điểm chính:**
- BFS frontier = khoảng cách ngắn nhất đến ô chưa thăm gần nhất (theo đường đi tự do)
- Giúp agent "biết hướng cần đi" thay vì random walk → học nhanh hơn đáng kể
- Nếu bỏ: agent phải học từ trial-error thuần tuý → cần nhiều bước huấn luyện hơn
- Đây là domain knowledge được encode vào observation — kỹ thuật phổ biến trong RL thực tế

---

**Q9.** Tại sao VDPPO dùng Tanh cho Actor nhưng ReLU cho AgentValueHead?
> ★★★

**Điểm chính:**
- Actor output logits → cần khoảng (-∞, +∞) linh hoạt → Tanh ổn định hơn ReLU ở hidden layers
- Tanh: gradient vanishing ít hơn ở vùng gần 0, phù hợp với orthogonal init
- AgentValueHead ước tính giá trị cá nhân (thường non-negative trong bài toán hợp tác) → ReLU đủ
- Thực tế: paper MAPPO (Yu et al. 2022) khuyên dùng Tanh cho Actor trong cooperative MARL

---

**Q10.** Tại sao lr_critic (6e-4) > lr_actor (2e-4) trong VDPPO?
> ★★☆

**Điểm chính:**
- Critic cần hội tụ trước → ước tính V(s) chính xác → GAE Advantage chính xác → gradient Actor ổn định
- Nếu Critic học chậm, Advantage noisy → Actor update không đúng hướng
- Đây là kỹ thuật phổ biến trong Actor-Critic: Critic learn rate cao hơn 2-3×
- Trong đề tài: 6e-4 / 2e-4 = 3× — phù hợp với thông lệ

---

**Q11.** Global state có 304 chiều được tính thế nào? Tại sao không dùng raw pixels?
> ★☆☆ (Dễ)

**Điểm chính:**
- 100 (coverage map 10×10, 0/1) + 100 (obstacle map) + 100 (visit count chuẩn hoá) + 2 (pos UAV-0) + 2 (pos UAV-1) = 304
- Raw pixels: tốn bộ nhớ, cần CNN, training phức tạp hơn nhiều
- Structured vector: compact, interpretable, đủ thông tin cho Critic

---

**Q12.** 3 loại va chạm được xử lý khác nhau như thế nào? Tại sao không gộp làm 1?
> ★☆☆

**Điểm chính:**
- Head-on (trực diện) và Cross (chéo): **đưa về vị trí cũ + phạt nặng** — ngăn chặn hoàn toàn
- Overlap (chồng lấn): **cho phép nhưng phạt nhẹ** — không thể ngăn hoàn toàn trong discrete env
- Tại sao khác nhau: Head-on/Cross là va chạm "cứng" (physically impossible); Overlap xảy ra khi 2 UAV cùng tiến đến 1 ô từ hướng khác — không thể detect trước

---

## 7.3 Câu hỏi về Kết quả thực nghiệm

---

**Q13.** MAPPO đạt 95.7% trên Simple và 97.9% trên Bottleneck nhưng chỉ 68.1% trên Mixed. Giải thích nguyên nhân.
> ★★★

**Điểm chính:**
- Simple: chướng ngại vật rải rác → dễ navigate → SharedActor học được
- Bottleneck: cấu trúc cứng (1 lối) → agent học 1 chiến lược rõ ràng (qua lối [4,4]) → stable
- Mixed: nhiều tường ngang tạo hành lang → 2 UAV cùng vào hành lang, không có cơ chế phân chia → thất bại
- Overlap=0.2 trên Mixed → không phải do va chạm; do 2 agent đi vào vùng chết (dead-end corridors)
- VDPPO giải quyết: diversity reward + higher collision penalty → agent phân tán → 97.9%

---

**Q14.** VDPPO đạt 97.8% nhưng PPO đơn cũng đạt 96.7%. Lợi ích thực sự của MARL ở đây là gì?
> ★★★ (Câu hỏi bẫy thường gặp!)

**Điểm chính:**
- Đây là câu hỏi quan trọng! Không phải "VDPPO chỉ tốt hơn 1.1%"
- Quan trọng hơn: VDPPO dùng **2 UAV** — tức là bao phủ được **cùng diện tích với thời gian ít hơn**
- PPO: 1 UAV, 10M bước, ~116-129 bước/episode để đạt 90% coverage
- VDPPO: 2 UAV, 3.5M bước, **và có thể mỗi UAV bay ít bước hơn** (vì phân chia vùng)
- Thực tiễn: 2 UAV cùng coverage = **giảm 50% thời gian** tuần tra → quan trọng trong ứng dụng thực

---

**Q15.** Kết quả có reproducible không? Làm sao đảm bảo kết quả không phải may mắn?
> ★★☆

**Điểm chính:**
- Fixed seed=42 cho tất cả thực nghiệm → có thể reproduce chính xác
- VDPPO: std=0.31% trên 5 checkpoint cuối → rất ổn định, không phải may mắn
- Đánh giá trên 5 episode mỗi bản đồ, 3 bản đồ khác nhau → 15 episodes/đánh giá
- Tiếc là chưa có: multiple random seeds (chạy 5 lần với seed khác nhau) → hướng cải thiện tốt

---

**Q16.** Tại sao không so sánh với QMIX hay thuật toán khác?
> ★★☆

**Điểm chính:**
- QMIX là value-based (Q-learning), không phải policy gradient → training loop hoàn toàn khác
- Thêm QMIX cần implement lại từ đầu — vượt phạm vi đồ án một môn học
- Đề tài đã có 4 thuật toán theo lộ trình rõ ràng (PPO→IPPO→MAPPO→VDPPO)
- Hướng phát triển: có thể so sánh với QMIX trong nghiên cứu sau

---

**Q17.** Tại sao IPPO Round 2 có Joint Coverage (57.7%) thấp hơn Round 1 (57.3%) dù cá nhân tốt hơn?
> ★★★ (Câu hỏi khó, cần phân tích sâu)

**Điểm chính:**
- Agent cá nhân tốt hơn không đồng nghĩa phối hợp tốt hơn
- Round 2: Agent-0 được train với Agent-1 từ Round 1 → học "tránh" Agent-1 → nhưng Agent-1 cũng đang thay đổi → non-stationary!
- Kết quả là cả 2 agent "tốt hơn cá nhân" nhưng strategy của họ không complement nhau
- Đây là hiện tượng điển hình của IPPO: convergence không đảm bảo phối hợp
- Round 3 mới tốt hơn vì đủ thời gian để cả 2 ổn định cùng nhau

---

## 7.4 Câu hỏi về Hạn chế và Hướng phát triển

---

**Q18.** Môi trường 2D grid 10×10 có đủ để kết luận cho UAV thực tế không?
> ★★☆

**Điểm chính:**
- Không hoàn toàn — đây là simplification có chủ đích (proof of concept)
- Grid 2D bỏ qua: physics UAV (quán tính, gió, pin), không gian 3D, camera angle
- Nhưng: kiến trúc VDPPO (CTDE + value decomp) là general → có thể adapt sang continuous env
- Bước tiếp theo: AirSim (Microsoft) hoặc Gazebo (ROS) cho simulation 3D thực hơn
- Kết luận trong đề tài đã nói rõ hạn chế này → HĐ đánh giá cao sự trung thực

---

**Q19.** Nếu có thêm 3 tháng, bạn sẽ cải thiện gì đầu tiên?
> ★☆☆ (Nhưng cần có câu trả lời rõ ràng)

**Điểm chính (chọn 2-3 điểm có sức thuyết phục):**
1. **Ablation study VDPPO**: Tắt từng thành phần (chỉ V_team, chỉ adaptive entropy, chỉ diversity reward) → đo đóng góp từng cơ chế — quan trọng nhất về mặt khoa học
2. **Multiple seeds**: Chạy 5 lần với seed khác nhau → tính mean ± std thực sự → kết quả đáng tin cậy hơn
3. **Mở rộng 3 UAV**: Test xem VDPPO có giữ được hiệu quả khi N tăng không
4. **Map lớn hơn** (15×15 hoặc 20×20): Test scalability

---

**Q20.** Thuật toán VDPPO của bạn có điểm yếu nào không?
> ★★☆ (Câu hỏi về critical thinking — rất quan trọng!)

**Điểm chính (nên tự phê bình trung thực):**
1. **Sample efficiency thấp hơn off-policy**: VDPPO on-policy → mỗi trajectory chỉ dùng 1 lần → cần nhiều bước hơn MADDPG (off-policy)
2. **Chưa có ablation study**: Chưa biết thành phần nào của VDPPO quan trọng nhất
3. **Chạy trên CPU**: Kaggle CPU → training chậm → không thể thử nhiều hyperparameter
4. **Chỉ 2 agent**: Chưa test với N>2 — có thể performance giảm khi nhiều agent hơn
5. **Mixed map MAPPO vẫn chưa giải thích đầy đủ**: Giả thuyết credit assignment, nhưng chưa có bằng chứng trực tiếp

---

## 7.5 Câu hỏi "bẫy" — Thường gặp nhất

---

**Q21.** Đây có phải là MARL thực sự không? Hay chỉ là 2 PPO đơn chạy song song?
> ★★★ (Câu hỏi kinh điển!)

**Điểm chính:**
- IPPO: Đúng là 2 PPO chạy song song — nhưng vẫn là MARL (phi tập trung)
- MAPPO và VDPPO: MARL thực sự — CentralCritic nhận global state, cập nhật Actor dựa trên phối hợp
- Bằng chứng: nếu chỉ là 2 PPO đơn, không thể giải thích tại sao MAPPO 95.7% trên Simple nhưng IPPO chỉ 75.8% — CentralCritic tạo ra sự khác biệt

---

**Q22.** Coverage 97.8% có nghĩa là 2 UAV bao phủ 97.8% diện tích trong 1 episode — vậy 2.2% còn lại là gì?
> ★☆☆

**Điểm chính:**
- 2.2% = ~2-3 ô trong 93-84 ô tự do tùy bản đồ
- Nguyên nhân: ô ở góc khuất (dead end), hoặc agent không có đủ bước (max_steps=600/800)
- VDPPO đạt 97.8% ổn định → gần như luôn đạt ≥97% threshold (thưởng hoàn thành)
- Cải thiện: tăng max_steps hoặc điều chỉnh reward để ưu tiên ô còn sót

---

**Q23.** Bạn dùng "đồng thời" (simultaneous) cho 2 UAV — có vấn đề gì với giả định này không?
> ★★☆

**Điểm chính:**
- Simultaneous action: cả 2 UAV quyết định và thực hiện cùng lúc → không có "lợi thế" của việc hành động trước
- Thực tế: UAV có độ trễ (latency), không hoàn toàn simultaneous
- Trong đề tài: xử lý bằng cách detect collision **sau khi** cả 2 di chuyển → đơn giản nhưng đủ dùng
- Một hướng nâng cao: sequential action (Stackelberg game) hoặc xét độ trễ liên lạc

---

**Q24.** Tại sao không dùng Self-Play để cải thiện IPPO?
> ★★★ (Câu hỏi nâng cao)

**Điểm chính:**
- Self-Play: huấn luyện agent chống lại các phiên bản cũ của chính nó → phổ biến trong game competitive
- Bài toán này là **cooperative** (hợp tác), không phải competitive → Self-Play không trực tiếp áp dụng
- Tuy nhiên: có thể dùng phiên bản MAPPO (Self-Imitation Learning): Agent-0 học từ trajectory của Agent-1 và ngược lại
- Thực tế đề tài dùng alternating training — tương tự nhưng đơn giản hơn

---

## 7.6 Câu hỏi về trình bày

---

**Q25.** Bạn tự đánh giá đóng góp mới của đề tài là gì so với MAPPO gốc (Yu et al. 2022)?
> ★★☆ (Bắt buộc phải trả lời tốt)

**Điểm chính:**
1. **Ứng dụng domain mới**: MAPPO gốc test trên SMAC, GRF, Hanabi — đề tài áp dụng cho CPP đa UAV trên grid đô thị
2. **VDPPO = MAPPO + Value Decomp + Adaptive Entropy + Diversity Reward**: Kết hợp 3 cải tiến không có trong paper gốc
3. **Phân tích Coordination Gap**: Định lượng hiện tượng agent cá nhân tốt nhưng phối hợp kém — đóng góp phân tích
4. **Thiết kế môi trường**: UAVPatrolEnvIPPO với BFS frontier, 3 loại collision, 14-thành-phần reward

---

*Tổng số câu hỏi: 25 câu, bao gồm từ dễ (★) đến khó (★★★)*  
*Lời khuyên: Nắm chắc Q1, Q3, Q4, Q5, Q13, Q14, Q21, Q25 — đây là những câu HĐ hay hỏi nhất*

---

## Tóm tắt nhanh — Bảng "Thuộc lòng" trước bảo vệ

| Câu hỏi | Câu trả lời 1 dòng |
|---------|-------------------|
| PPO là gì? | Policy gradient on-policy với clipped objective để giới hạn bước cập nhật |
| CTDE là gì? | Train tập trung (Critic dùng global state), Deploy phân tán (Actor dùng local obs) |
| Coordination Gap là gì? | Cá nhân 91% nhưng chung 69% — gap 22% do 2 agent đi trùng vùng |
| Tại sao MAPPO thất bại Mixed? | Hành lang hẹp + SharedActor không phân chia credit cá nhân → không phân vùng được |
| VDPPO khác MAPPO ở đâu? | V_total = V_team + mean(V_agent) — gán credit chính xác hơn cho từng agent |
| Kết quả tốt nhất? | VDPPO: 97.8% TB (Simple 97.8%, Mixed 97.9%, Bottleneck 97.6%), overlap 1.3 |
| Seed thực nghiệm? | 42 (cố định, reproducible) |
| Chạy ở đâu? | Kaggle CPU, PyTorch + SB3 + Gymnasium |
| Số bước VDPPO? | 3.5 triệu bước, 8 env song song, hội tụ tại ~update 210 |

---

*👤 Lê Đức Chiến (2251172253) — 64KTPM3 — Đại học Thuỷ Lợi — 2026*  
*📁 https://github.com/chienday/uav-marl-patrol*
