# Assignment 4 - Generative Adversarial Networks (GANs)

**Course:** DS552 - Generative AI  
**Module:** 4 - Generative Adversarial Networks
**Name:** Piyush Thapar

---

## Overview

This assignment explores Generative Adversarial Networks (GANs), a powerful framework where two neural networks (generator and discriminator) compete in a minimax game to produce realistic synthetic data. We implement DCGAN-style architectures in PyTorch on both MNIST and CIFAR-10 datasets.

---

## Files in This Directory

| File | Description |
|------|-------------|
| `Assignment_4_Coding.ipynb` | Main assignment notebook with theory + coding tasks |
| `040---Module-4---Generative-Adversarial-Networks.pdf` | Course lecture notes on GANs |
| `041---Assignment-4---Geenerative-Adversarial-Networks.pdf` | Assignment instructions |
| `generated_images_mnist/` | Saved MNIST sample grids (every 10 epochs) |
| `generated_images_cifar10/` | Saved CIFAR-10 sample grids (every 10 epochs) |

---

## Jupyter Notebook Contents

### `Assignment_4_Coding.ipynb`

The notebook is organized into two main parts:

---

### Part 1: Theory Questions

Four comprehensive answers to fundamental GAN concepts:

| Question | Topic |
|----------|-------|
| **Q1** | Explain the minimax loss function and how it ensures competitive training |
| **Q2** | What is mode collapse? Why does it occur and how can it be mitigated? |
| **Q3** | Explain the role of the discriminator in adversarial training |
| **Q4** | How do metrics like IS and FID evaluate GAN performance? |

---

### Part 2: Coding Tasks

**Note:** The original assignment code was provided in TensorFlow. We re-implement everything in **PyTorch** for consistency with our course environment.

#### Section A: GAN on MNIST (28×28×1)

- **Objective:** Implement a basic DCGAN to generate handwritten digits
- **Dataset:** MNIST (60,000 training images, 28×28×1)
- **Architecture:**
  - **Generator:** Linear → 7×7×256 → 3 ConvTranspose2d blocks → 28×28×1
  - **Discriminator:** 2 Conv2d blocks with BatchNorm → Flatten → Linear(1)
- **Epochs:** 50
- **Image saving:** Every 10 epochs to `generated_images_mnist/`

---

#### Section B: GAN on CIFAR-10 (32×32×3)

- **Objective:** Replace MNIST with CIFAR-10 and add additional convolutional layers
- **Dataset:** CIFAR-10 (50,000 training images, 32×32×3)
- **Architecture:**
  - **Generator:** Linear → 4×4×512 → **4 ConvTranspose2d blocks** (additional layers) → 32×32×3
  - **Discriminator:** 3 Conv2d blocks with BatchNorm → Flatten → Linear(1)
- **Epochs:** 100 (CIFAR-10 is harder, needs more training)
- **Image saving:** Every 10 epochs to `generated_images_cifar10/`

---

## Hyperparameters Used (DCGAN Best Practices)

| Parameter | MNIST | CIFAR-10 |
|-----------|-------|----------|
| Learning Rate | 2e-4 | 2e-4 |
| Adam Betas | (0.5, 0.999) | (0.5, 0.999) |
| Noise Dimension | 100 | 100 |
| Batch Size | 256 | 128 |
| Epochs | 50 | 100 |
| Label Smoothing (real) | 0.9 | 0.9 |
| Weight Init | N(0, 0.02) | N(0, 0.02) |

---

## Tuning Improvements Applied

| Technique | Description |
|-----------|-------------|
| **DCGAN Weight Init** | Conv/Linear ~ N(0, 0.02), BatchNorm ~ N(1, 0.02) — critical for convergence |
| **Adam Betas (0.5, 0.999)** | Lower beta1 reduces momentum, prevents oscillation in adversarial training |
| **Label Smoothing** | Real labels = 0.9 instead of 1.0 — prevents discriminator overconfidence |
| **BatchNorm in Discriminator** | On hidden layers (not first) — stabilizes training per DCGAN paper |
| **Generator uses ReLU** | DCGAN recommends ReLU in generator, LeakyReLU only in discriminator |
| **`drop_last=True`** | Avoids batch size mismatch in last batch |

---

## Key Concepts Demonstrated

### Minimax Loss Function
```
min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

### Binary Cross-Entropy Loss (Implementation)
```python
# Discriminator loss
d_loss = BCE(D(real), 1) + BCE(D(G(z)), 0)

# Generator loss
g_loss = BCE(D(G(z)), 1)
```

### Training Alternation
```
1. Train Discriminator: maximize classification accuracy (real vs fake)
2. Train Generator: minimize discriminator's ability to detect fakes
```

---

## How to Run

1. **Install Dependencies:**
   ```bash
   pip install torch torchvision matplotlib numpy
   ```

2. **Open the Notebook:**
   ```bash
   jupyter notebook Assignment_4_Coding.ipynb
   ```

3. **Run All Cells:**
   - Theory questions are in markdown cells at the top
   - Section A trains on MNIST (~10–15 minutes with GPU for 50 epochs)
   - Section B trains on CIFAR-10 (~30–60 minutes with GPU for 100 epochs)
   - Generated images are saved to `generated_images_mnist/` and `generated_images_cifar10/`

---

## Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision
- matplotlib
- numpy
- CUDA (recommended for faster training)

---

## Key Takeaways

1. **Weight initialization matters:** DCGAN-style N(0, 0.02) init is essential for stable GAN training
2. **Adam betas (0.5, 0.999):** Lower momentum (beta1=0.5) prevents oscillation in adversarial optimisation
3. **Label smoothing:** Using 0.9 for real labels keeps the discriminator from becoming too confident
4. **CIFAR-10 is harder:** Requires more capacity (additional conv layers), smaller batches, and more epochs
5. **Mode collapse awareness:** Monitoring both G and D losses helps detect training instability

---

## References

- Goodfellow et al. (2014). "Generative Adversarial Nets"
- Radford et al. (2016). "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (DCGAN)
- Course Materials: Module 4 - Generative Adversarial Networks
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
