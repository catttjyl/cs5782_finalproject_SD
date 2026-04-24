# cs5482_finalproject_SD

### Reproduction of "Deep Networks with Stochastic Depth" (Huang et al., 2016)

---

Our code implements a 110-layer ResNet with stochastic depth training on CIFAR-10.

Paper: arXiv:1603.09382v3
Original code (Torch 7): https://github.com/yueatsprograms/Stochastic_Depth

#### Usage:

Train with stochastic depth

```
python stochastic_depth.py --mode stochastic
```

Train baseline (constant depth)

```
python stochastic_depth.py --mode constant
```

Train both and compare

```
python stochastic_depth.py --mode both
```
