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
│   └── evaluation/             🚧 In Progress
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
|---------|:------------:|:-----------:|
| ✅ ON  | > 0          | **Strong**  |
| ❌ OFF | 0            | Weaker      |

### ✔ Key Findings

- Semantic signal is not auxiliary, but **causal**
- Control mechanism actively **modifies decisions**
- Gate remains **non-trivial** (non-collapsed)

---

## ⚠️ Current Status

| Component | Status |
|-----------|--------|
| Model Implementation | ✅ Complete |
| Training Pipeline | ✅ Complete |
| Mechanism Validation (ON vs OFF) | ✅ Complete |
| Real Dataset Experiments | 🚧 In Progress |
| Evaluation Module | 🚧 In Progress |

---

## 🧪 Upcoming Experiments

- [ ] MovieLens-1M
- [ ] Amazon Books
- [ ] Recall@K / NDCG@K evaluation
- [ ] Cold-start analysis
- [ ] Sparsity robustness study

---

## 🧠 Contribution

🔥 SCA introduces a new perspective:

| Paradigm | Approach |
|----------|----------|
| ❌ Traditional | Embedding fusion |
| ❌ Common LLM4Rec | Feature augmentation |
| ✅ **SCA** | **Decision-level control learning** |

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

## 🔥 Contribution Streak

<div align="center">

<img src="https://github-readme-streak-stats.herokuapp.com/?user=appleweiping&theme=tokyonight&hide_border=true"/>

</div>

---

## ⏱️ Wakatime Coding Stats

<!--
  前置步骤（一次性配置，之后全自动）：
  1. 注册 https://wakatime.com 并在 VS Code / PyCharm 安装插件
  2. 仓库 Settings → Secrets and variables → Actions → 新建 WAKATIME_API_KEY
  3. 在本仓库新建文件 .github/workflows/waka-readme.yml，内容如下：

  ────────────────────────────────────────
  name: Waka Readme
  on:
    schedule:
      - cron: "0 0 * * *"
    workflow_dispatch:
  jobs:
    update-readme:
      runs-on: ubuntu-latest
      steps:
        - uses: anmol098/waka-readme-stats@master
          with:
            WAKATIME_API_KEY: ${{ secrets.WAKATIME_API_KEY }}
            GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
            SHOW_LINES_OF_CODE: true
            SHOW_PROFILE_VIEWS: false
            SHOW_COMMIT: true
            SHOW_LANGUAGE: true
            SHOW_OS: true
  ────────────────────────────────────────

  4. 配置后手动触发一次 workflow，下方标记区域将自动填充真实数据
-->

<!--START_SECTION:waka-->
<!--END_SECTION:waka-->

---

<div align="center">

**If you find this interesting or useful, consider giving it a ⭐!**

[![Star History](https://api.star-history.com/svg?repos=appleweiping/SCA-Recommendation&type=Date)](https://star-history.com/#appleweiping/SCA-Recommendation&Date)

![footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer&animation=twinkling)

</div>