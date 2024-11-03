# Adversarial Training
>Adversarial training involves a two-player game between the adversary and the defender to train models with adversarial examples to improve model resilence.

- [1 Definition](#1-Definition)
- [2 Overfitting (Generalization)](#2-Overfitting-Generalization)
  - [2.1 Catastrophic Overfitting](#21-Catastrophic-Overfitting)
    - [2.1.1 Underlying Reasons](#211-Underlying-Reasons)
    - [2.1.2 Solutions](#212-Solutions)  
  - [2.2 Robust Overfitting](#22-Robust-Overfitting)
    - [2.2.1 Underlying Reasons](#221-Underlying-Reasons)
    - [2.2.2 Solutions](#222-Solutions)  
- [3 Adversarial Robustness Enhancement](#3-Adversarial-Robustness-Enhancement)
- [4 Robust Fairness](#4-Robust-Fairness)
- [5 Trade-off between Adversarial Robustness and Standard Accuracy](#5-Trade-off-between-Adversarial-Robustness-and-Standard-Accuracy)
- [6 Comparison and Connection between Adversarial Training and Randomized Smoothing](#6-Comparison-and-Connection-between-Adversarial-Training-and-Randomized-Smoothing)
- [7 Defense against Patch Attack](#7-Defense-against-Patch-Attack)
- [8 Multi-Attack Robustness](#8-Multi-Attack-Robustness)
- [9 Cross-network and task Adversarial Training](#9-Cross-network-and-task-Adversarial-Training)
- [10 Robust Pre-training and Fine-tuning](#10-Robust-Pre-training-and-Fine-tuning)
- [11 Adaptive Perturbations](#11-Adaptive-Perturbations)
- [12 Efficiency](#12-Efficiency)
- [13 Adversarial Training for ViTs and Comparison with CNNs](#13-Adversarial-Training-for-ViTs-and-Comparison-with-CNNs)
- [14 Adversarial Training against Poisoning Attack](#14-Adversarial-Training-against-Poisoning-Attack)
  - [14.1 Adversarial Training against Backdoor Attack](#141-Adversarial-Training-against-Backdoor-Attack)
  - [14.2 Adversarial-Training-against-Availability-Attack](#142-Adversarial-Training-against-Availability-Attack)
  <!-- - [Citation](#citation) -->

# Adversarial Training

## 1 Definition
- [2017] **Towards deep learning models resistant to adversarial attacks.** [[paper](https://arxiv.org/abs/1706.06083)]
- [2018] **Adversarial logit pairing.** [[paper](https://arxiv.org/abs/1803.06373)]

## 2 Overfitting (Generalization)

### 2.1 Catastrophic Overfitting
##### 2.1.1 Underlying Reasons
- [2024] **Improving fast adversarial training with prior-guided knowledge.** [[paper](https://ieeexplore.ieee.org/abstract/document/10478545/)]
- [2024] **Eliminating catastrophic overfitting via abnormal adversarial examples regularization.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d65befe6b80ecf7f180b4def503d7776-Abstract-Conference.html)]
- [2024] **Taxonomy driven fast adversarial training.** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28330)]
- [2024] **Revisiting single-step adversarial training for robustness and generalization.** [[paper](https://www.sciencedirect.com/science/article/pii/S0031320324001079)]
- [2023] **Fast adversarial training with adaptive step size.** [[paper](https://ieeexplore.ieee.org/abstract/document/10298035/)]
- [2023] **Investigating catastrophic overfitting in fast adversarial training: A self-fitting perspective.** [[paper](https://openaccess.thecvf.com/content/CVPR2023W/AML/html/He_Investigating_Catastrophic_Overfitting_in_Fast_Adversarial_Training_A_Self-Fitting_Perspective_CVPRW_2023_paper.html)]
- [2023] **Catastrophic overfitting can be induced with discriminative non-robust features.** [[paper](https://openreview.net/forum?id=10hCbu70Sr)]
- [2023] **On the over-memorization during natural, robust and catastrophic overfitting.** [[paper](https://arxiv.org/abs/2310.08847)]
- [2023] **Efficient local linearity regularization to overcome catastrophic overfitting.** [[paper](https://arxiv.org/abs/2401.11618)]
- [2023] **The enemy of my enemy is my friend: Exploring inverse adversaries for improving adversarial training.** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Dong_The_Enemy_of_My_Enemy_Is_My_Friend_Exploring_Inverse_CVPR_2023_paper.html)]
- [2023] **Fast adversarial training with smooth convergence.** [[paper](http://openaccess.thecvf.com/content/ICCV2023/html/Zhao_Fast_Adversarial_Training_with_Smooth_Convergence_ICCV_2023_paper.html)]
- [2022] **Subspace adversarial training.** [[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Subspace_Adversarial_Training_CVPR_2022_paper)]
- [2022] **Frequencylowcut pooling-plug and play against catastrophic overfitting.** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-19781-9_3)]
- [2022] **Boosting fast adversarial training with learnable adversarial initialization.** [[paper](https://ieeexplore.ieee.org/abstract/document/9807638/)]
- [2022] **Prior-guided adversarial initialization for fast adversarial training.** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-19772-7_33)]
- [2021] **Reliably fast adversarial training via latent adversarial perturbation.** [[paper](http://openaccess.thecvf.com/content/ICCV2021/html/Park_Reliably_Fast_Adversarial_Training_via_Latent_Adversarial_Perturbation_ICCV_2021_paper.html)]
- [2021] **Understanding catastrophic overfitting in single-step adversarial training.** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16989)]
- [2020] **Single-step adversarial training with dropout scheduling.** [[paper](https://ieeexplore.ieee.org/abstract/document/9157154/)]
- [2020] **Understanding and improving fast adversarial training.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2020/hash/b8ce47761ed7b3b6f48b583350b7f9e4-Abstract.html)]
- [2019] **Fast is better than free: Revisiting adversarial training.** [[paper](https://arxiv.org/abs/2001.03994)]
- [2017] **Towards deep learning models resistant to adversarial attacks.** [[paper](https://arxiv.org/abs/1706.06083)]
##### 2.1.2 Solutions
- [2022] **Make some noise: Reliable and efficient single-step adversarial training.** [[paper]([https](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5434a6b40f8f65488e722bc33d796c8b-Abstract-Conference.html))]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

### 2.2 Robust Overfitting
##### 2.2.1 Underlying Reasons

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
##### 2.2.2 Solutions
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

## 3 Adversarial Robustness Enhancement
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
## 4 Robust Fairness
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
## 5 Trade-off between Adversarial Robustness and Standard Accuracy
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
## 6 Comparison and Connection between Adversarial Training and Randomized Smoothing
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
## 7 Defense against Patch Attack
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
## 8 Multi-Attack Robustness
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
## 9 Cross-network and task Adversarial Training
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
## 10 Robust Pre-training and Fine-tuning
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
## 11 Adaptive Perturbations
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
## 12 Efficiency
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
## 13 Adversarial Training for ViTs and Comparison with CNNs
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
## 14 Adversarial Training against Poisoning Attack
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
### 14.1 Adversarial Training against Backdoor Attack
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
### 14.2 Adversarial Training against Availability Attack
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]

- [2023] **Adapting.** [[paper](https)]
- [2023] **Adapting.** [[paper](https)]
