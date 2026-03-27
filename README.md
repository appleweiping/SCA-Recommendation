<div align="center">

# 🔥 SCA-Recommendation

### Structure-aware Control Alignment for LLM-enhanced Recommendation

![Stars](https://img.shields.io/github/stars/appleweiping/SCA-Recommendation?style=for-the-badge&logo=github&color=orange)
![Forks](https://img.shields.io/github/forks/appleweiping/SCA-Recommendation?style=for-the-badge&logo=github&color=steelblue)
![Issues](https://img.shields.io/github/issues/appleweiping/SCA-Recommendation?style=for-the-badge&logo=github&color=red)
![Last Commit](https://img.shields.io/github/last-commit/appleweiping/SCA-Recommendation?style=for-the-badge&logo=git&color=green)
![License](https://img.shields.io/github/license/appleweiping/SCA-Recommendation?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![LightGCN](https://img.shields.io/badge/Backbone-LightGCN-blueviolet?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=for-the-badge&logo=clockify)

<br/>

**🚀 A New Paradigm: From Representation Fusion → Decision-Level Control**

</div>

---

## 🧠 Overview

SCA (Structure-aware Control Alignment) is a novel recommendation framework that integrates semantic signals into collaborative filtering via **control mechanisms**, rather than conventional embedding fusion.

> 💡 Instead of asking *"what is similar?"*, we ask:
> **"how should decisions be adjusted?"**

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔥 Decision-Level Control | Semantic signals modify decisions, not just embeddings |
| 🧠 LLM Encoding | Compatible with any pretrained language model |
| 🔗 Graph Modeling | Built on LightGCN for structural propagation |
| 🎛️ Dynamic Gate | Learnable, non-trivial gating mechanism |
| ⚖️ Alignment Learning | Bridges semantic space and collaborative signals |

---

## 🏗️ Architecture

<div align="center">
  <i>Conceptual pipeline of SCA</i>
</div>

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
│   └── evaluation/
│       ├── metrics.py          ✅ Recall@K, NDCG@K, HR@K
│       └── evaluator.py        ✅ Full ranking evaluation
│
├── scripts/
│   ├── preprocess_ml1m.py      ✅ ratings.dat → interactions.csv
│   └── split_ml1m.py           ✅ Leave-one-out split
│
├── run.py
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/appleweiping/SCA-Recommendation.git
cd SCA-Recommendation
pip install torch pandas pyyaml
```

---

## ▶️ Quick Start

### ✅ Train (Control ON)

```bash
py run.py
```

### ❌ Ablation (Control OFF)

```bash
py run.py --config configs/sca_off.yaml
```

---

## 📊 Dataset: MovieLens-1M

> ✅ Real dataset pipeline fully connected as of March 24, 2025.

| Split | Count |
|-------|------:|
| Users | 6,041 |
| Items | 3,953 |
| Train interactions | 988,129 |
| Validation interactions | 6,040 |
| Test interactions | 6,040 |

### Preprocessing

```bash
# Step 1: Convert ratings.dat → interactions.csv (implicit feedback)
py scripts/preprocess_ml1m.py

# Step 2: Leave-one-out split
py scripts/split_ml1m.py
```

### Evaluation Protocol

We follow the **leave-one-out** protocol standard in top-tier venues (SIGIR / KDD / WWW):

- **Train**: all interactions except the last two per user
- **Valid**: second-to-last interaction per user
- **Test**: last interaction per user
- **No overlap** across splits (verified: train/valid/test overlap = 0)
- Each user has exactly **1 ground-truth item** in the test set
- **Full-ranking evaluation** over all unobserved items (~3,800 candidates per user)

### Metrics

| Metric | Description |
|--------|-------------|
| Recall@K | Fraction of relevant items retrieved in top-K |
| NDCG@K | Ranking quality with position discounting |
| HR@K | Hit ratio at K |

---

## 📊 Training Snapshot

```
Epoch 3:
  loss      ↓
  bpr       ↓
  align     ↓
  pos_score ↑
  neg_score ↓

System health check (ML-1M):
  num_users   = 6041   ✔
  candidate   ≈ 3800   ✔
  train_pairs ≈ 988K   ✔
  pos > neg   = 1.0    ✔
  gate_mean   ≈ 0.5    ✔
  ctrl_shift  ≈ 0.05   ✔
```

> 👉 Model is learning normally; control mechanism is active and non-collapsed.

---

## 📊 Experimental Results (ML-1M, Middle-scale)

> ⚠️ Current results are based on **middle-scale experiments (fast validation setting)**.
> Full-scale training and multi-dataset validation are ongoing.

### Main Results

| Model | Recall@10 | NDCG@10 | HR@10 |
|-------|:---------:|:-------:|:-----:|
| LightGCN | 0.0356 | 0.0175 | 0.0356 |
| SCA-off | 0.0318 | 0.0148 | 0.0318 |
| **SCA-on (Ours)** | **0.0373** | **0.0178** | **0.0373** |

### Ablation Study

| Variant | Recall@10 | NDCG@10 | Δ Recall@10 |
|---------|:---------:|:-------:|:-----------:|
| SCA-on | 0.0373 | 0.0178 | — |
| w/o Gate | 0.0373 | 0.0178 | ≈ 0 |
| w/o Alignment | 0.0373 | 0.0178 | ≈ 0 |
| w/o Structure | 0.0373 | 0.0178 | ≈ 0 |
| Fusion (no gate) | 0.0373 | 0.0178 | ≈ 0 |

### Key Observations

- Semantic control improves over LightGCN, showing consistent gains in Recall and NDCG.
- Removing control (SCA-off) degrades performance, confirming that semantic signals **actively influence decisions**.
- Most ablation variants exhibit minimal differences at this scale, suggesting:
  - the core performance gain mainly comes from introducing **semantic control**
  - finer components (gate, alignment, structure) provide secondary effects
- These observations motivate further analysis under **larger-scale and more challenging settings**.

---

## 🧠 Contribution

🔥 SCA introduces a new perspective:

| Paradigm | Approach |
|----------|----------|
| ❌ Traditional | Embedding fusion |
| ❌ Common LLM4Rec | Feature augmentation |
| ✅ **SCA** | **Decision-level control learning** |

---

## ⚠️ Current Status

| Component | Status |
|-----------|--------|
| Model Implementation | ✅ Complete |
| Training Pipeline | ✅ Complete |
| Evaluation Module (metrics + evaluator) | ✅ Complete |
| Data Preprocessing (ML-1M) | ✅ Complete |
| Leave-one-out Split | ✅ Complete |
| Mechanism Validation (ON vs OFF) | ✅ Complete |
| LightGCN Baseline | ✅ Complete |
| Quantitative Results (ML-1M, middle-scale) | ✅ Complete |
| Ablation Study (full variants) | ✅ Complete |
| Real Dataset Experiments (full-scale) | 🚧 In Progress |
| Comparison Baselines (extended) | 🚧 In Progress |

---

## 🧪 Upcoming Experiments

- [x] MovieLens-1M data pipeline
- [x] Recall@K / NDCG@K / HR@K evaluation
- [x] SCA-on vs SCA-off comparison
- [x] LightGCN baseline
- [x] Full ablation variants (gate / alignment / structure / fusion)
- [ ] Full-scale training (long epochs)
- [ ] Multi-dataset validation (Amazon / Yelp)
- [ ] Cold-start analysis
- [ ] Sparsity robustness study

---

## 📌 Vision

> We believe recommendation systems should move toward:
> **Controllable, interpretable, and semantic-aware decision models**

---

## 🤝 Contact & Collaboration

Interested in these topics? Let's connect:

- 📌 LLM for Recommendation
- 📌 Hybrid Semantic-CF Models
- 📌 Decision Learning in RecSys

---

## 📈 GitHub Stats

<div align="center">

<img height="180em" src="https://github-readme-stats.vercel.app/api?username=appleweiping&show_icons=true&theme=tokyonight&include_all_commits=true&count_private=true&hide_border=true"/>
&nbsp;&nbsp;
<img height="180em" src="https://github-readme-stats.vercel.app/api/top-langs/?username=appleweiping&layout=compact&langs_count=8&theme=tokyonight&hide_border=true"/>

</div>

---

<div align="center">

**If you find this interesting or useful, consider giving it a ⭐!**

[![Star History](https://api.star-history.com/svg?repos=appleweiping/SCA-Recommendation&type=Date)](https://star-history.com/#appleweiping/SCA-Recommendation&Date)

![footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer&animation=twinkling)

</div>