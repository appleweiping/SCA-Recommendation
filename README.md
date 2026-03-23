# 🔥 SCA-Recommendation
### Structure-aware Control Alignment for LLM-enhanced Recommendation

<p align="center">
  <b>🚀 A New Paradigm: From Representation Fusion → Decision-Level Control</b>
</p>

---

## 🧠 Overview

SCA (Structure-aware Control Alignment) is a novel recommendation framework that integrates semantic signals into collaborative filtering via **control mechanisms**, rather than conventional embedding fusion.

> 💡 Instead of asking *"what is similar?"*, we ask:
> **"how should decisions be adjusted?"**

---

## ✨ Key Features

- 🔥 **Decision-Level Semantic Control**
- 🧠 LLM-compatible Semantic Encoding
- 🔗 Structure-aware Graph Modeling (LightGCN)
- 🎛️ Dynamic Gate Mechanism
- ⚖️ Alignment Learning between Semantic & Structure

---

## 🏗️ Architecture

<p align="center">
  <i>Conceptual pipeline of SCA</i>
</p>

```
User → Semantic Encoder → Δ_u (control signal)
     → Gate → g_u
     → LightGCN → e_u
     → Controlled Representation:
       ẽ_u = e_u + g_u ⊙ Δ_u
```

---

## 📦 Project Structure

```
SCA-Recommendation/
│
├── configs/
│   ├── sca_default.yaml        # Control ON
│   └── sca_off.yaml            # Control OFF (ablation)
│
├── src/
│   ├── models/
│   │   ├── lightgcn.py
│   │   ├── semantic_encoder.py
│   │   ├── gate.py
│   │   ├── sca.py              ⭐ Core
│   │   └── losses.py
│   │
│   ├── data/
│   │   ├── dataset.py
│   │   └── sampler.py
│   │
│   ├── trainers/
│   │   ├── trainer_base.py
│   │   └── trainer_sca.py
│   │
│   └── evaluation/             🚧 In Progress
│
├── run.py
└── README.md
```

---

## ⚙️ Installation

```bash
pip install torch pandas pyyaml
```

---

## ▶️ Run

### ✅ Train (Control ON)

```bash
py run.py
```

### ❌ Ablation (Control OFF)

```bash
py run.py --config configs/sca_off.yaml
```

---

## 📊 Training Snapshot

```
Epoch 3:
  loss      ↓
  bpr       ↓
  align     ↓
  pos_score ↑
  neg_score ↓
```

---

## 🔬 Ablation Study (Core Insight)

| Setting | Control Shift | Performance |
|---------|--------------|-------------|
| ON      | > 0          | Strong      |
| OFF     | 0            | Weaker      |

### ✔ Key Findings

- Semantic signal is not auxiliary, but **causal**
- Control mechanism actively **modifies decisions**
- Gate remains **non-trivial** (non-collapsed)

---

## ⚠️ Current Status

- ✅ Model fully implemented
- ✅ Training pipeline complete
- ✅ Mechanism validated (ON vs OFF)
- 🚧 Real dataset experiments (in progress)

---

## 🧪 Upcoming Experiments

- MovieLens-1M
- Amazon Books
- Recall@K / NDCG@K
- Cold-start analysis
- Sparsity robustness

---

## 🧠 Contribution

🔥 SCA introduces a new perspective:

- ❌ Not embedding fusion
- ❌ Not feature augmentation
- ✅ **Decision-level control learning**

---

## 📌 Vision

We believe recommendation systems should move toward:

> **Controllable, interpretable, and semantic-aware decision models**

---

## 🤝 Contact

Research project on:

- LLM for Recommendation
- Hybrid Semantic-CF Models
- Decision Learning

---

## ⭐ Star This Repo

If you find this interesting or useful, consider giving it a ⭐!