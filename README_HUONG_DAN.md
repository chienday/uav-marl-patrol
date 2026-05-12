# 📚 HƯỚNG DẪN ĐỌC — UAV MARL Patrol

> **Lê Đức Chiến (2251172253)** — 64KTPM3 — Đại học Thuỷ Lợi — 2026  
> Đề tài: *Tuần tra đô thị bằng đội UAV sử dụng Học Tăng Cường Đa Tác Tử (MARL)*  
> Kết quả nổi bật: **VDPPO đạt 97.8% coverage** — vượt PPO đơn tác tử với 2 UAV phối hợp

---

## Tôi nên bắt đầu từ đâu?

```
Bạn là ai?
    │
    ├─► Chưa biết RL bao giờ          → 🟢 Lộ trình A (~15 giờ)
    ├─► Biết AI/ML, chưa chuyên sâu RL → 🟡 Lộ trình B (~8 giờ)
    └─► Đã biết RL/MARL                → 🔴 Lộ trình C (~3 giờ)
```

---

## 🟢 Lộ trình A — Người mới hoàn toàn

| Bước | Đọc / Làm | Mục đích | Thời gian |
|:----:|-----------|---------|:---------:|
| 1 | `GUIDE_DATN.md` Phần 1 (Lý thuyết từ zero) | Hiểu RL bằng ví dụ, không cần toán nặng | 3-4 giờ |
| 2 | `GUIDE_DATN.md` Phần 2 (Cài đặt) | Chạy được code trên máy hoặc Kaggle | 1 giờ |
| 3 | `notebooks/ppo-final.ipynb` | Xem PPO 1 UAV học thế nào | 1.5 giờ |
| 4 | `GUIDE_DATN.md` Phần 4 (Đọc mã nguồn) | Hiểu cấu trúc file, không cần đọc hết | 1.5 giờ |
| 5 | `notebooks/vdppo-final.ipynb` | Xem thuật toán tốt nhất của đề tài | 2 giờ |
| 6 | `Draft_DATN.md` Chương 1–3 | Đọc bài toán, môi trường | 1.5 giờ |
| 7 | `Draft_DATN.md` Chương 5–6 | Đọc kết quả và phân tích | 1.5 giờ |
| 8 | `GUIDE_DATN.md` Phần 7 (Câu hỏi HĐ) | Chuẩn bị buổi bảo vệ | 3 giờ |

---

## 🟡 Lộ trình B — Biết AI/ML, chưa sâu về RL

| Bước | Đọc / Làm | Mục đích | Thời gian |
|:----:|-----------|---------|:---------:|
| 1 | `GUIDE_DATN.md` Phần 1.4→1.8 (PPO → MARL) | Bỏ qua ML cơ bản, học thẳng RL policy gradient | 2 giờ |
| 2 | `Draft_DATN.md` Chương 2 (Cơ sở lý thuyết) | Công thức đầy đủ, kiến trúc mạng | 1.5 giờ |
| 3 | `src/envs/uav_env_multi.py` + `reward.py` | Hiểu môi trường và hàm thưởng | 1 giờ |
| 4 | `src/agents/networks.py` + `vdppo_trainer.py` | Đọc kiến trúc VDPPO — **quan trọng nhất** | 1.5 giờ |
| 5 | `Draft_DATN.md` Chương 5–6 | Kết quả thực nghiệm với bảng số liệu thực | 1 giờ |
| 6 | `GUIDE_DATN.md` Phần 7 (Câu hỏi HĐ) | Chuẩn bị bảo vệ | 2 giờ |

---

## 🔴 Lộ trình C — Đã biết RL/MARL

| Bước | Đọc / Làm | Mục đích | Thời gian |
|:----:|-----------|---------|:---------:|
| 1 | `Draft_DATN.md` Chương 3–4 (Môi trường + Kiến trúc) | Nắm thiết kế hệ thống | 45 phút |
| 2 | `src/agents/vdppo_trainer.py` + `networks.py` | Implementation chi tiết | 45 phút |
| 3 | `logs/vdppo_experiment/eval_history.json` | Phân tích dữ liệu thô | 30 phút |
| 4 | `Draft_DATN.md` Chương 6 (Thảo luận) | Coordination Gap + phân tích MAPPO failure | 30 phút |
| 5 | `GUIDE_DATN.md` Phần 5–7 | Mở rộng + Câu hỏi nâng cao của HĐ | 1 giờ |

---

## 🗂️ Sơ đồ phụ thuộc tài liệu

```
📄 GUIDE_DATN.md              ← ĐỌC ĐẦU TIÊN nếu mới
   (Lý thuyết + Code + Câu hỏi HĐ)
           │
           ▼
📄 Draft_DATN.md               ← Bản thảo đồ án hoàn chỉnh (7 chương)
           │
      ┌────┴────────────┐
      ▼                 ▼
📓 notebooks/        🗺️ maps/
   ppo-final.ipynb       map_simple.json      (7 chướng ngại)
   ippo-final.ipynb      map_mixed.json       (16 chướng ngại, hành lang)
   mappo-final.ipynb     map_bottleneck.json  (chỉ 1 lối thông ở [4,4])
   vdppo-final.ipynb
           │
      ┌────┴──────────────────────────┐
      ▼                               ▼
🐍 src/envs/                    🐍 src/agents/
   uav_env_single.py (PPO)          ippo_trainer.py  → SB3 alternating
   uav_env_multi.py  (MARL) ──►     mappo_trainer.py → custom PyTorch
   reward.py (14 thành phần)        vdppo_trainer.py → MAPPO + VD
                                    networks.py ★ SharedActor/Critic/Head
                                          │
                               📊 logs/*/eval_history.json
                               📊 logs/*/config.json
                               📈 plots/ (PNG biểu đồ)
                               💾 checkpoints/ (model .pt/.zip)
```

---

## 📋 Tóm tắt từng file quan trọng

### Tài liệu
| File | Mô tả ngắn | Đọc khi nào |
|------|-----------|------------|
| `Draft_DATN.md` | Bản thảo đồ án 7 chương, 84KB — lý thuyết→kết quả→kết luận | Cần bức tranh tổng thể |
| `GUIDE_DATN.md` | Hướng dẫn đầy đủ: lý thuyết zero-to-hero + code + câu hỏi HĐ | Cần học hoặc chuẩn bị bảo vệ |

### Notebooks (chạy trên Kaggle)
| File | Mô tả | Kết quả |
|------|-------|---------|
| `ppo-final.ipynb` | PPO 1 UAV, SB3, 10M bước, MLP [512-256-128] | 96.7% TB |
| `ippo-final.ipynb` | IPPO 2 UAV độc lập, 3 vòng × 2M bước | 69.6% TB (Joint) |
| `mappo-final.ipynb` | MAPPO CTDE, SharedActor + CentralCritic | 87.2% TB |
| `vdppo-final.ipynb` | VDPPO = MAPPO + Value Decomp + Adaptive Entropy | **97.8% TB** ★ |

### Mã nguồn (src/)
| File | Mô tả | Độ khó |
|------|-------|:------:|
| `src/envs/uav_env_multi.py` | Môi trường 2 UAV: step/reset/collision/BFS frontier | ★★☆ |
| `src/envs/reward.py` | 14 thành phần reward, 3 config khác nhau | ★★☆ |
| `src/agents/networks.py` | **SharedActor / CentralCritic / AgentValueHead** — trái tim | ★★★ |
| `src/agents/vdppo_trainer.py` | Rollout → GAE → PPO Clip → Value Decomp update | ★★★ |
| `src/agents/mappo_trainer.py` | Giống VDPPO nhưng không có AgentValueHead | ★★★ |
| `src/agents/ippo_trainer.py` | SingleAgentWrapper + SB3 PPO alternating training | ★★☆ |

### Dữ liệu
| File | Mô tả |
|------|-------|
| `maps/*.json` | Định nghĩa bản đồ JSON — dễ đọc, dễ tạo mới |
| `logs/*/config.json` | Siêu tham số thực tế của từng lần chạy |
| `logs/*/eval_history.json` | Lịch sử coverage/overlap/entropy theo từng update |
| `checkpoints/vdppo/*.pt` | Trọng số VDPPO đã train — load ngay không cần train lại |

---

## ⚡ Quick Start — Chạy ngay trong 5 bước

```bash
# Bước 1 — Clone về
git clone https://github.com/chienday/uav-marl-patrol.git
cd uav-marl-patrol

# Bước 2 — Cài thư viện
pip install torch>=2.0 gymnasium>=0.29 stable-baselines3[extra]>=2.0 numpy matplotlib

# Bước 3 — Xem demo VDPPO đã train (không cần GPU, không cần train lại)
# Mở notebooks/vdppo-final.ipynb → chạy phần Evaluation ở cuối

# Bước 4 — Train thử VDPPO (test nhanh, 500K bước thay vì 3.5M)
python -c "
from src.envs import UAVPatrolEnvIPPO, VDPPORewardConfig
from src.agents import VDPPOTrainer
trainer = VDPPOTrainer(map_file='maps/map_simple.json',
    map_paths={'simple':'maps/map_simple.json',
               'mixed':'maps/map_mixed.json',
               'bottleneck':'maps/map_bottleneck.json'},
    total_steps=500_000)
trainer.train()
"

# Bước 5 — Tạo biểu đồ từ log
python scripts/generate_plots.py
```

---

## 📖 9 Paper cần biết (theo thứ tự ưu tiên)

| # | Tên | Tác giả | Năm | Link |
|---|-----|---------|:---:|------|
| 1 | **MAPPO** *(nền tảng trực tiếp)* | Yu et al. | 2022 | [arXiv:2103.01955](https://arxiv.org/abs/2103.01955) |
| 2 | **VDN** *(nền tảng value decomp)* | Sunehag et al. | 2018 | [arXiv:1706.05296](https://arxiv.org/abs/1706.05296) |
| 3 | **PPO** *(thuật toán gốc)* | Schulman et al. | 2017 | [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) |
| 4 | **QMIX** *(so sánh value decomp)* | Rashid et al. | 2018 | [arXiv:1803.11485](https://arxiv.org/abs/1803.11485) |
| 5 | **MADDPG** *(CTDE off-policy)* | Lowe et al. | 2017 | [arXiv:1706.02275](https://arxiv.org/abs/1706.02275) |
| 6 | **GAE** *(kỹ thuật advantage)* | Schulman et al. | 2015 | [arXiv:1506.02438](https://arxiv.org/abs/1506.02438) |
| 7 | **CPP Survey** *(bối cảnh bài toán)* | Galceran & Carreras | 2013 | [DOI](https://doi.org/10.1016/j.robot.2013.09.004) |
| 8 | **Multi-UAV MARL** *(ứng dụng gần nhất)* | Zhang et al. | 2024 | [DOI](https://doi.org/10.1088/1742-6596/2833/1/012017) |
| 9 | **Dec-POMDP** *(framework lý thuyết)* | Oliehoek & Amato | 2016 | [Springer](https://link.springer.com/book/10.1007/978-3-319-28929-8) |

---

## 🔑 Số liệu cần thuộc lòng trước buổi bảo vệ

```
┌─────────────────────────────────────────────────────────────────┐
│  BẢNG KẾT QUẢ TÓM TẮT — Coverage trung bình 3 bản đồ          │
├─────────────┬──────┬───────┬──────┬──────────┬──────┬─────────┤
│ Thuật toán  │ UAV  │ Bước  │Simple│  Mixed   │ BN   │   TB    │
├─────────────┼──────┼───────┼──────┼──────────┼──────┼─────────┤
│ PPO         │  1   │  10M  │ 94.6%│  97.8%   │ 97.8%│  96.7%  │
│ IPPO        │  2   │   6M  │ 75.8%│  65.6%   │ 67.5%│  69.6%  │
│ MAPPO       │  2   │  ~4M  │ 95.7%│  68.1% ✗ │ 97.9%│  87.2%  │
│ VDPPO       │  2   │  3.5M │ 97.8%│  97.9% ✓ │ 97.6%│  97.8% ★│
└─────────────┴──────┴───────┴──────┴──────────┴──────┴─────────┘

Coordination Gap (IPPO): Cá nhân ~91% nhưng Chung chỉ 69% → gap ~22%
VDPPO Overlap = 1.3  |  VDPPO Entropy = 0.33  |  Hội tụ tại Update ~120
```

---

*📁 Repo: https://github.com/chienday/uav-marl-patrol*  
*👤 Lê Đức Chiến (2251172253) — 64KTPM3 — Đại học Thuỷ Lợi — 2026*
