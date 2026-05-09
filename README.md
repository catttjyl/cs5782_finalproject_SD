# Deep Networks with Stochastic Depth (Replication Study)

**CS 5782 Final Project** | Jin Fan (jf936) · Yixuan Yang (yy2445) · Youlun Jiang (yj622)

---

## Introduction

This repository contains a replication study of [Huang et al., "Deep Networks with Stochastic Depth" (arXiv:1603.09382, 2016)](https://arxiv.org/abs/1603.09382). The paper proposes randomly dropping entire residual blocks during training and bypassing them with identity connections, resolving the tension between training efficiency and test-time expressiveness in very deep ResNets.

---

## Chosen Result

We reproduce **Table 1** of the original paper: a 110-layer ResNet trained with stochastic depth on CIFAR-10 achieves ~5.25% test error vs. ~6.41% for constant depth, with ~25% faster training. This result is the paper's central empirical claim, directly validating SD as both a regularizer and a training accelerator.

---

## GitHub Contents

```
├── code/         # Re-implementation code and training scripts
├── data/         # Dataset instructions (CIFAR-10)
├── results/      # Generated figures, tables, and logs
├── poster/       # Final poster PDF
├── report/       # Final 2-page report PDF
└── README.md
```

---

## Re-implementation Details

- **Model:** 110-layer ResNet (3 × 18 blocks, filters: 16 / 32 / 64), PyTorch
- **Dataset:** CIFAR-10 (45k train / 5k val / 10k test)
- **Training:** SGD, momentum 0.9, weight decay 1e-4, lr=0.1 decayed 10× at epochs 250 and 375, 500 epochs total
- **Survival probabilities:** linear decay, pL = 0.5
- **Device:** A100 GPU
- **Key modification:** Original code in Torch 7; we re-implemented fully in PyTorch

---

## Reproduction Steps

```bash
# 1. Clone the repo
git clone https://github.com/catttjyl/cs5482_finalproject_SD
cd cs5482_finalproject_SD

# 2. Install dependencies
pip install torch torchvision matplotlib scikit-learn

```

**Compute requirement:** A single A100 GPU; ~2.5 hours per run.

---

## Results / Insights

| Method | Test Error | Training Time |
|---|---|---|
| Constant Depth | 6.13% | 2h 51m |
| Stochastic Depth | 5.60% | 2h 28m (−13.2%) |

Beyond accuracy, stochastic depth maintains stronger gradient flow in early layers, learns more generalizable feature representations (higher k-NN accuracy despite looser clusters), and is significantly more robust to label noise.

---

## Conclusion

Stochastic depth is an effective and simple technique that improves accuracy, reduces training time, alleviates vanishing gradients, and resists overfitting. The replication confirms the paper's main claims, with the speedup gap attributable to modern GPU efficiency.

---

## References

- [1] G. Huang, Y. Sun, Z. Liu, D. Sedra, and K. Q. Weinberger, “Deep networks with stochastic depth,” arXiv:1603.09382, 2016.

---

## Acknowledgements

This project was completed as part of **CS 5782: Intro to Deep Learning** at Cornell University, Spring 2026.
