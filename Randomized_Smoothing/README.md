
## Randomized Smoothing

>Randomized smoothing (RS) aims to ensure certified adversarial robustness against evasion attacks (in most cases), which means that no future adversary (under a certain threat model) will break the defense.

>The paper list will be continuously updated to keep track of the latest papers.

  - [1 Basic Learning Methods](#1-Basic-Learning-Methods)
  - [2 Diverse Perturbations](#2-Diverse-Perturbations)
    - [2.1 Beyond l_2-norm Robustness Certification](#21-Beyond-l_2-norm-Robustness-Certification)
    - [2.2 Beyond Gaussian Noise](#22-Beyond-Gaussian-Noise)
- [3 Certified Robustness and its Trade-off with Standard Accuracy](#3-Certified-Robustness-and-its-Trade-off-with-Standard-Accuracy)
  - [4 Efficiency](#4-Efficiency)
  - [5 Curse of Dimensionality](#5-Curse-of-Dimensionality)
  - [6 Comparison and Connection between Adversarial Training and Randomized Smoothing](#6-Comparison-and-Connection-between-Adversarial-Training-and-Randomized-Smoothing)
  - [7 Defend against Non-adversarial Perturbation-based Evasion Attack](#7-Defend-against-Non-adversarial-Perturbation-based-Evasion-Attack)
  <!-- - [Citation](#citation) -->
  
#  PAPER LIST

## 1 Basic Learning Methods
- [2019] **Certified robustness to adversarial examples with differential privacy.** [[paper](https://ieeexplore.ieee.org/abstract/document/8835364/)]
- [2019] **Certified adversarial robustness via randomized smoothing.** [[paper](https://proceedings.mlr.press/v97/cohen19c.html)]
## 2 Diverse Perturbations
### 2.1 Beyond l_2-norm Robustness Certification
- [2023] **Certified adversarial robustness within multiple perturbation bounds.** [[paper](https://openaccess.thecvf.com/content/CVPR2023W/AML/html/Nandi_Certified_Adversarial_Robustness_Within_Multiple_Perturbation_Bounds_CVPRW_2023_paper.html)]
- [2022] **Gsmooth: Certified robustness against semantic transformations via generalized randomized smoothing.** [[paper](https://proceedings.mlr.press/v162/hao22c)]
- [2021] **Tss: Transformation-specific smoothing for robustness certification.** [[paper](https://dl.acm.org/doi/abs/10.1145/3460120.3485258)]
- [2020] **Wasserstein smoothing: Certified robustness against wasserstein adversarial attacks.** [[paper](http://proceedings.mlr.press/v108/levine20a.html)]
- [2020] **Robustness certificates for sparse adversarial attacks by randomized ablation.** [[paper](https://aaai.org/ojs/index.php/AAAI/article/view/5888)]
- [2019] **Certified adversarial robustness with additive noise.** [[paper](https://proceedings.neurips.cc/paper/2019/hash/335cd1b90bfa4ee70b39d08a4ae0cf2d-Abstract.html)]
- [2019] **Certified robustness to adversarial examples with differential privacy.** [[paper](https://ieeexplore.ieee.org/abstract/document/8835364/)]
- [2019] **Tight certificates of adversarial robustness for randomly smoothed classifiers.** [[paper](https://proceedings.neurips.cc/paper/2019/hash/fa2e8c4385712f9a1d24c363a2cbe5b8-Abstract.html)]
### 2.2 Beyond Gaussian Noise
- [2023] **Certified adversarial robustness within multiple perturbation bounds.** [[paper](https://openaccess.thecvf.com/content/CVPR2023W/AML/html/Nandi_Certified_Adversarial_Robustness_Within_Multiple_Perturbation_Bounds_CVPRW_2023_paper.html)]
- [2023] **Dimension-independent certified neural network watermarks via mollifier smoothing.** [[paper](https://proceedings.mlr.press/v202/ren23c.html)]
- [2022] **Gsmooth: Certified robustness against semantic transformations via generalized randomized smoothing.** [[paper](https://proceedings.mlr.press/v162/hao22c)]
- [2021] **Tss: Transformation-specific smoothing for robustness certification.** [[paper](https://dl.acm.org/doi/abs/10.1145/3460120.3485258)]
- [2020] **Robustness certificates for sparse adversarial attacks by randomized ablation.** [[paper](https://aaai.org/ojs/index.php/AAAI/article/view/5888)]
- [2020] **Randomized smoothing of all shapes and sizes.** [[paper](http://proceedings.mlr.press/v119/yang20c.html)]
- [2020] **Wasserstein smoothing: Certified robustness against wasserstein adversarial attacks.** [[paper](http://proceedings.mlr.press/v108/levine20a.html)]
- [2019] **Certified adversarial robustness with additive noise.** [[paper](https://proceedings.neurips.cc/paper/2019/hash/335cd1b90bfa4ee70b39d08a4ae0cf2d-Abstract.html)]
- [2019] **Tight certificates of adversarial robustness for randomly smoothed classifiers.** [[paper](https://proceedings.neurips.cc/paper/2019/hash/fa2e8c4385712f9a1d24c363a2cbe5b8-Abstract.html)]
- [2019] **Certified robustness to adversarial examples with differential privacy.** [[paper](https://ieeexplore.ieee.org/abstract/document/8835364/)]
## 3 Certified Robustness and its Trade-off with Standard Accuracy
- [2024] **Multi-scale diffusion denoised smoothing.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d51e2a4628b15518f58bd1056b2d9124-Abstract-Conference.html)]
- [2024] **Hierarchical randomized smoothing.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9c0efc0d84c263972af72bf70a2de533-Abstract-Conference.html)]
- [2024] **Adversarial examples might be avoidable: The role of data concentration in adversarial robustness.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/92d21245424f3898b7110f555a00e829-Abstract-Conference.html)]
- [2023] **{DiffSmooth}: Certifiably robust learning via diffusion models and local smoothing.** [[paper](https://www.usenix.org/conference/usenixsecurity23/presentation/zhang-jiawei)]
- [2022] **(Certified!!) Adversarial robustness for free!.** [[paper](https://arxiv.org/abs/2206.10550)]
- [2022] **Data dependent randomized smoothing.** [[paper](https://proceedings.mlr.press/v180/alfarra22a)]
- [2019] **Certified adversarial robustness via randomized smoothing.** [[paper](https://proceedings.mlr.press/v97/cohen19c.html)]
- [2019] **Certified adversarial robustness with additive noise.** [[paper](https://proceedings.neurips.cc/paper/2019/hash/335cd1b90bfa4ee70b39d08a4ae0cf2d-Abstract.html)]
- [2020] **Consistency regularization for certified robustness of smoothed classifiers.** [[paper](https://proceedings.neurips.cc/paper/2020/hash/77330e1330ae2b086e5bfcae50d9ffae-Abstract.html)]
- [2023] **Certified adversarial robustness within multiple perturbation bounds.** [[paper](https://openaccess.thecvf.com/content/CVPR2023W/AML/html/Nandi_Certified_Adversarial_Robustness_Within_Multiple_Perturbation_Bounds_CVPRW_2023_paper.html)]
- [2019] **Provably robust deep learning via adversarially trained smoothed classifiers.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2019/hash/3a24b25a7b092a252166a1641ae953e7-Abstract.html)]
- [2019] **MACER: Attack-free and scalable robust training via maximizing certified radius.** [[paper](https://arxiv.org/abs/2001.02378)]

## 4 Efficiency
- [2023] **Incremental randomized smoothing certification.** [[paper](https://arxiv.org/abs/2305.19521)]
## 5 Curse of Dimensionality
- [2020] **Randomized smoothing of all shapes and sizes.** [[paper](http://proceedings.mlr.press/v119/yang20c.html)]
## 6 Comparison and Connection between Adversarial Training and Randomized Smoothing
- [2024] **DRF: Improving certified robustness via distributional robustness framework.** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/29504)]
- [2023] **Certified distributional robustness on smoothed classifiers.** [[paper](https://ieeexplore.ieee.org/abstract/document/10093123/)]
- [2017] **Certifying some distributional robustness with principled adversarial training.** [[paper](https://arxiv.org/abs/1710.10571)]
- [2019] **Provably robust deep learning via adversarially trained smoothed classifiers.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2019/hash/3a24b25a7b092a252166a1641ae953e7-Abstract.html)]

## 7 Defend against Non-adversarial Perturbation-based Evasion Attack
- [2022] **Gsmooth: Certified robustness against semantic transformations via generalized randomized smoothing.** [[paper](https://proceedings.mlr.press/v162/hao22c)]
- [2021] **Tss: Transformation-specific smoothing for robustness certification.** [[paper](https://dl.acm.org/doi/abs/10.1145/3460120.3485258)]
- [2020] **Certified defense to image transformations via randomized smoothing.** [[paper](https://proceedings.neurips.cc/paper/2020/hash/5fb37d5bbdbbae16dea2f3104d7f9439-Abstract.html)]
