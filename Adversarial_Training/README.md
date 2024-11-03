# Adversarial Training
>Adversarial training involves a two-player game between the adversary and the defender to train models with adversarial examples to improve model resilience.
>The paper list will be continuously updated to keep track of the latest papers.

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

# PAPER LIST

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
- [2024] **Improving fast adversarial training with prior-guided knowledge.** [[paper](https://ieeexplore.ieee.org/abstract/document/10478545/)]
- [2024] **Eliminating catastrophic overfitting via abnormal adversarial examples regularization.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d65befe6b80ecf7f180b4def503d7776-Abstract-Conference.html)]
- [2024] **Taxonomy driven fast adversarial training.** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28330)]
- [2024] **Revisiting single-step adversarial training for robustness and generalization.** [[paper](https://www.sciencedirect.com/science/article/pii/S0031320324001079)]
- [2023] **Efficient local linearity regularization to overcome catastrophic overfitting.** [[paper](https://arxiv.org/abs/2401.11618)]
- [2023] **On the over-memorization during natural, robust and catastrophic overfitting.** [[paper](https://arxiv.org/abs/2310.08847)]
- [2023] **Investigating catastrophic overfitting in fast adversarial training: A self-fitting perspective.** [[paper](https://openaccess.thecvf.com/content/CVPR2023W/AML/html/He_Investigating_Catastrophic_Overfitting_in_Fast_Adversarial_Training_A_Self-Fitting_Perspective_CVPRW_2023_paper.html)]
- [2023] **The enemy of my enemy is my friend: Exploring inverse adversaries for improving adversarial training.** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Dong_The_Enemy_of_My_Enemy_Is_My_Friend_Exploring_Inverse_CVPR_2023_paper.html)]
- [2023] **Fast adversarial training with adaptive step size.** [[paper](https://ieeexplore.ieee.org/abstract/document/10298035/)]
- [2023] **Fast adversarial training with smooth convergence.** [[paper](http://openaccess.thecvf.com/content/ICCV2023/html/Zhao_Fast_Adversarial_Training_with_Smooth_Convergence_ICCV_2023_paper.html)]
- [2023] **Towards stable and efficient adversarial training against l_1 bounded adversarial attacks.** [[paper](https://proceedings.mlr.press/v202/jiang23f.html)]
- [2022] **Subspace adversarial training.** [[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Subspace_Adversarial_Training_CVPR_2022_paper)]
- [2022] **Prior-guided adversarial initialization for fast adversarial training.** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-19772-7_33)]
- [2022] **Frequencylowcut pooling-plug and play against catastrophic overfitting.** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-19781-9_3)]
- [2022] **Make some noise: Reliable and efficient single-step adversarial training.** [[paper]([https](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5434a6b40f8f65488e722bc33d796c8b-Abstract-Conference.html))]
- [2022] **Boosting fast adversarial training with learnable adversarial initialization.** [[paper](https://ieeexplore.ieee.org/abstract/document/9807638/)]
- [2021] **Understanding catastrophic overfitting in single-step adversarial training.** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16989)]
- [2021] **Reliably fast adversarial training via latent adversarial perturbation.** [[paper](http://openaccess.thecvf.com/content/ICCV2021/html/Park_Reliably_Fast_Adversarial_Training_via_Latent_Adversarial_Perturbation_ICCV_2021_paper.html)]
- [2020] **Understanding and improving fast adversarial training.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2020/hash/b8ce47761ed7b3b6f48b583350b7f9e4-Abstract.html)]
- [2020] **Single-step adversarial training with dropout scheduling.** [[paper](https://ieeexplore.ieee.org/abstract/document/9157154/)]

### 2.2 Robust Overfitting
##### 2.2.1 Underlying Reasons
- [2024] **Regional adversarial training for better robust generalization.** [[paper](https://link.springer.com/article/10.1007/s11263-024-02103-w)]
- [2024] **Balance, imbalance, and rebalance: Understanding robust overfitting from a minimax game perspective.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/32f9049217da6e718a426b07242dff73-Abstract-Conference.html)]
- [2023] **A3T: Accuracy aware adversarial training.** [[paper](https://link.springer.com/article/10.1007/s10994-023-06341-w)]
- [2023] **Mitigating robust overfitting via self-residual-calibration regularization.** [[paper](https://www.sciencedirect.com/science/article/pii/S0004370223000231)]
- [2023] **Understanding and combating robust overfitting via input loss landscape analysis and regularization.** [[paper](https://www.sciencedirect.com/science/article/pii/S0031320322007087)]
- [2023] **Exploring the relationship between architectural design and adversarially robust generalization.** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Liu_Exploring_the_Relationship_Between_Architectural_Design_and_Adversarially_Robust_Generalization_CVPR_2023_paper.html)]
- [2022] **CAT: Customized adversarial training for improved robustness.** [[paper](https://arxiv.org/abs/2002.06789)]
- [2022] **Understanding robust overfitting of adversarial training and beyond.** [[paper](https://proceedings.mlr.press/v162/yu22b.html)]
- [2021] **Robust overfitting may be mitigated by properly learned smoothening.** [[paper](https://openreview.net/forum?id=qZzy5urZw9)]
- [2021] **Exploring memorization in adversarial training.** [[paper](https://arxiv.org/abs/2106.01606)]
- [2021] **Low curvature activations reduce overfitting in adversarial training.** [[paper](http://openaccess.thecvf.com/content/ICCV2021/html/Singla_Low_Curvature_Activations_Reduce_Overfitting_in_Adversarial_Training_ICCV_2021_paper.html)]
- [2021] **Fixing data augmentation to improve adversarial robustness.** [[paper](https://arxiv.org/abs/2103.01946)]
- [2020] **Adversarial vertex mixup: Toward better adversarially robust generalization.** [[paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Lee_Adversarial_Vertex_Mixup_Toward_Better_Adversarially_Robust_Generalization_CVPR_2020_paper.html)]
- [2020] **Confidence-calibrated adversarial training: Generalizing to unseen attacks.** [[paper](https://proceedings.mlr.press/v119/stutz20a.html)]
- [2020] **Geometry-aware instance-reweighted adversarial training.** [[paper](https://arxiv.org/abs/2010.01736)]
- [2020] **Adversarial weight perturbation helps robust generalization.** [[paper](https://proceedings.neurips.cc/paper/2020/hash/1ef91c212e30e14bf125e9374262401f-Abstract.html?ref=https://githubhelp.com)]
- [2019] **Adversarial robustness may be at odds with simplicity.** [[paper](https://arxiv.org/abs/1901.00532)]
- [2019] **Robust local features for improving the generalization of adversarial training.** [[paper](https://arxiv.org/abs/1909.10147)]
- [2018] **Adversarially robust generalization requires more data.** [[paper](https://proceedings.neurips.cc/paper/2018/hash/f708f064faaf32a43e4d3c784e6af9ea-Abstract.html)]
- [2018] **Averaging weights leads to wider optima and better generalization.** [[paper](https://arxiv.org/abs/1803.05407)]
  
##### 2.2.2 Solutions

- [2024] **Regional adversarial training for better robust generalization.** [[paper](https://link.springer.com/article/10.1007/s11263-024-02103-w)]
- [2024] **Balance, imbalance, and rebalance: Understanding robust overfitting from a minimax game perspective.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/32f9049217da6e718a426b07242dff73-Abstract-Conference.html)]
- [2023] **Boosting adversarial robustness via self-paced adversarial training.** [[paper](https://www.sciencedirect.com/science/article/pii/S0893608023004938)]
- [2023] **A3T: Accuracy aware adversarial training.** [[paper](https://link.springer.com/article/10.1007/s10994-023-06341-w)]
- [2023] **Understanding and combating robust overfitting via input loss landscape analysis and regularization.** [[paper](https://www.sciencedirect.com/science/article/pii/S0031320322007087)]
- [2023] **Mitigating robust overfitting via self-residual-calibration regularization.** [[paper](https://www.sciencedirect.com/science/article/pii/S0004370223000231)]
- [2023] **Interpolated joint space adversarial training for robust and generalizable defenses.** [[paper](https://ieeexplore.ieee.org/abstract/document/10155464/)]
- [2023] **Self-ensemble adversarial training for improved robustness.** [[paper](https://arxiv.org/abs/2203.09678)]
- [2023] **Better diffusion models further improve adversarial training.** [[paper](https://proceedings.mlr.press/v202/wang23ad.html)]
- [2022] **Consistency regularization for adversarial robustness.** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20817)]
- [2022] **CAT: Customized adversarial training for improved robustness.** [[paper](https://arxiv.org/abs/2002.06789)]
- [2022] **Understanding robust overfitting of adversarial training and beyond.** [[paper](https://proceedings.mlr.press/v162/yu22b.html)]
- [2022] **Data augmentation alone can improve adversarial training.** [[paper](https://arxiv.org/abs/2301.09879)]
- [2021] **Low curvature activations reduce overfitting in adversarial training.** [[paper](http://openaccess.thecvf.com/content/ICCV2021/html/Singla_Low_Curvature_Activations_Reduce_Overfitting_in_Adversarial_Training_ICCV_2021_paper.html)]
- [2021] **Exploring memorization in adversarial training.** [[paper](https://arxiv.org/abs/2106.01606)]
- [2021] **Robust overfitting may be mitigated by properly learned smoothening.** [[paper](https://openreview.net/forum?id=qZzy5urZw9)]
- [2021] **Improving robustness using generated data.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/21ca6d0cf2f25c4dbb35d8dc0b679c3f-Abstract.html)]
- [2021] **Fixing data augmentation to improve adversarial robustness.** [[paper](https://arxiv.org/abs/2103.01946)]
- [2021] **Data augmentation can improve robustness.** [[paper](https://proceedings.neurips.cc/paper/2021/hash/fb4c48608ce8825b558ccf07169a3421-Abstract.html)]
- [2020] **Adversarial weight perturbation helps robust generalization.** [[paper](https://proceedings.neurips.cc/paper/2020/hash/1ef91c212e30e14bf125e9374262401f-Abstract.html?ref=https://githubhelp.com)]
- [2020] **Geometry-aware instance-reweighted adversarial training.** [[paper](https://arxiv.org/abs/2010.01736)]
- [2020] **Adversarial vertex mixup: Toward better adversarially robust generalization.** [[paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Lee_Adversarial_Vertex_Mixup_Toward_Better_Adversarially_Robust_Generalization_CVPR_2020_paper.html)]
- [2020] **Confidence-calibrated adversarial training: Generalizing to unseen attacks.** [[paper](https://proceedings.mlr.press/v119/stutz20a.html)]
- [2019] **Improving adversarial robustness requires revisiting misclassified examples.** [[paper](https://openreview.net/forum?id=rklOg6EFwS)]
- [2019] **On the convergence and robustness of adversarial training.** [[paper](https://arxiv.org/abs/2112.08304)]
- [2019] **Adversarially robust generalization just requires more unlabeled data.** [[paper](https://arxiv.org/abs/1906.00555)]
- [2019] **Using pre-training can improve model robustness and uncertainty.** [[paper](https://proceedings.mlr.press/v97/hendrycks19a.html)]
- [2019] **Robust local features for improving the generalization of adversarial training.** [[paper](https://arxiv.org/abs/1909.10147)]
- [2019] **Are labels required for improving adversarial robustness?** [[paper](https://proceedings.neurips.cc/paper_files/paper/2019/hash/bea6cfd50b4f5e3c735a972cf0eb8450-Abstract.html)]
- [2018] **Averaging weights leads to wider optima and better generalization.** [[paper](https://arxiv.org/abs/1803.05407)]
- [2018] **Adversarially robust generalization requires more data.** [[paper](https://proceedings.neurips.cc/paper/2018/hash/f708f064faaf32a43e4d3c784e6af9ea-Abstract.html)]
- [2018] **Curriculum adversarial training.** [[paper](https://arxiv.org/abs/1805.04807)]


## 3 Adversarial Robustness Enhancement

- [2024] **Exploring robust features for improving adversarial robustness.** [[paper](https://ieeexplore.ieee.org/abstract/document/10495132/)]
- [2024] **Defense against adversarial attacks using topology aligning adversarial training.** [[paper](https://ieeexplore.ieee.org/abstract/document/10416271/)]
- [2024] **Improving adversarial robustness via information bottleneck distillation.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/233278d812e74a4f9848410881db86b1-Abstract-Conference.html)]
- [2023] **Improving adversarial robustness with self-paced hard-class pair reweighting.** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26738)]
- [2023] **Edge enhancement improves adversarial robustness in image classification.** [[paper](https://www.sciencedirect.com/science/article/pii/S092523122201342X)]
- [2023] **Feature separation and recalibration for adversarial robustness.** [[paper](
http://openaccess.thecvf.com/content/CVPR2023/html/Kim_Feature_Separation_and_Recalibration_for_Adversarial_Robustness_CVPR_2023_paper.html)]
- [2023] **Theoretically grounded loss functions and algorithms for adversarial robustness.** [[paper](https://proceedings.mlr.press/v206/awasthi23c.html)]
- [2022] **Enhancing adversarial training with second-order statistics of weights.** [[paper](http://openaccess.thecvf.com/content/CVPR2022/html/Jin_Enhancing_Adversarial_Training_With_Second-Order_Statistics_of_Weights_CVPR_2022_paper.html)]
- [2021] **Self-ensemble adversarial training for improved robustness.** [[paper](https://arxiv.org/abs/2203.09678)]
- [2021] **Cifs: Improving adversarial robustness of cnns via channel-wise importance-based feature selection.** [[paper](https://proceedings.mlr.press/v139/yan21e.html)]
- [2020] **Adversarial self-supervised contrastive learning.** [[paper](https://proceedings.neurips.cc/paper/2020/hash/1f1baa5b8edac74eb4eaa329f14a0361-Abstract.html)]
- [2020] **Improving adversarial robustness via channel-wise activation suppressing.** [[paper](https://arxiv.org/abs/2103.08307)]
- [2019] **Improving adversarial robustness via promoting ensemble diversity.** [[paper](http://proceedings.mlr.press/v97/pang19a)]
- [2019] **Feature denoising for improving adversarial robustness.** [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Xie_Feature_Denoising_for_Improving_Adversarial_Robustness_CVPR_2019_paper.html)]
- [2019] **Metric learning for adversarial robustness.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2019/hash/c24cd76e1ce41366a4bbe8a49b02a028-Abstract.html)]
- [2017] **Ensemble adversarial training: Attacks and defenses.** [[paper](https://arxiv.org/abs/1705.07204)]

## 4 Robust Fairness

- [2024] **DAFA: Distance-aware fair adversarial training.** [[paper](https://arxiv.org/abs/2401.12532)]
- [2023] **Combining adversaries with anti-adversaries in training.** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26352)]
- [2023] **Improving robust fairness via balance adversarial training.** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26769)]
- [2023] **Cfa: Class-wise calibrated fair adversarial training.** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Wei_CFA_Class-Wise_Calibrated_Fair_Adversarial_Training_CVPR_2023_paper.html)]
- [2021] **Robustness may be at odds with fairness: An empirical study on class-wise accuracy.** [[paper](https://proceedings.mlr.press/v148/benz21a)]
- [2021] **Analysis and applications of class-wise robustness in adversarial training.** [[paper](https://dl.acm.org/doi/abs/10.1145/3447548.3467403)]
- [2021] **To be robust or to be fair: Towards fairness in adversarial training.** [[paper](http://proceedings.mlr.press/v139/xu21b.html)]

## 5 Trade-off between Adversarial Robustness and Standard Accuracy




- [2024] **Maximization of average precision for deep learning with adversarial ranking robustness.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/31f04c174a6af322e9417b7a9a91097a-Abstract-Conference.html)]
- [2024] **Enhance adversarial robustness via geodesic distance.** [[paper](https://ieeexplore.ieee.org/abstract/document/10431598/)]
- [2024] **Revisiting the trade-off between accuracy and robustness via weight distribution of filters.** [[paper](https://ieeexplore.ieee.org/abstract/document/10552117/)]
- [2024] **Attention-based investigation and solution to the trade-off issue of adversarial training.** [[paper](https://www.sciencedirect.com/science/article/pii/S0893608024001485)]
- [2024] **Connecting certified and adversarial training.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e8b0c97b34fdaf58b2f48f8cca85e76a-Abstract-Conference.html)]
- [2023] **Interpolated joint space adversarial training for robust and generalizable defenses.** [[paper](https://ieeexplore.ieee.org/abstract/document/10155464/)]
- [2023] **The enemy of my enemy is my friend: Exploring inverse adversaries for improving adversarial training.** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Dong_The_Enemy_of_My_Enemy_Is_My_Friend_Exploring_Inverse_CVPR_2023_paper.html)]
- [2023] **Combining adversaries with anti-adversaries in training.** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26352)]
- [2023] **One-vs-the-rest loss to focus on important samples in adversarial training.** [[paper](https://proceedings.mlr.press/v202/kanai23a.html)]
- [2023] **Towards desirable decision boundary by moderate-margin adversarial training.** [[paper](https://proceedings.mlr.press/v119/zhang20z.html)]
- [2023] **Adversarial robustness via random projection filters.** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Dong_Adversarial_Robustness_via_Random_Projection_Filters_CVPR_2023_paper.html)]
- [2023] **Push stricter to decide better: A class-conditional feature adaptive framework for improving adversarial robustness.** [[paper](https://ieeexplore.ieee.org/abstract/document/10098264/)]
- [2023] **Improving adversarial robustness by learning shared information.** [[paper](https://www.sciencedirect.com/science/article/pii/S0031320322005349)]
- [2023] **Randomized adversarial training via taylor expansion.** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Jin_Randomized_Adversarial_Training_via_Taylor_Expansion_CVPR_2023_paper.html)]
- [2023] **Float: Fast learnable once-for-all adversarial training for tunable trade-off between accuracy and robustness.** [[paper](https://openaccess.thecvf.com/content/WACV2023/html/Kundu_FLOAT_Fast_Learnable_Once-for-All_Adversarial_Training_for_Tunable_Trade-Off_Between_WACV_2023_paper.html)]
- [2023] **Improving generalization of adversarial training via robust critical fine-tuning.** [[paper](http://openaccess.thecvf.com/content/ICCV2023/html/Zhu_Improving_Generalization_of_Adversarial_Training_via_Robust_Critical_Fine-Tuning_ICCV_2023_paper.html)]
- [2023] **Conserve-update-revise to cure generalization and robustness trade-off in adversarial training.** [[paper](https://arxiv.org/abs/2401.14948)]
- [2023] **Generalist: Decoupling natural and robust generalization.** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Wang_Generalist_Decoupling_Natural_and_Robust_Generalization_CVPR_2023_paper.html)]
- [2023] **Advancing example exploitation can alleviate critical challenges in adversarial training.** [[paper](http://openaccess.thecvf.com/content/ICCV2023/html/Ge_Advancing_Example_Exploitation_Can_Alleviate_Critical_Challenges_in_Adversarial_Training_ICCV_2023_paper.html)]
- [2021] **Robust overfitting may be mitigated by properly learned smoothening.** [[paper](https://openreview.net/forum?id=qZzy5urZw9)]
- [2021] **Improving robustness using generated data.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/21ca6d0cf2f25c4dbb35d8dc0b679c3f-Abstract.html)]
- [2021] **Reducing excessive margin to achieve a better accuracy vs. robustness trade-off.** [[paper](https://openreview.net/forum?id=Azh9QBQ4tR7)]
- [2021] **Fast AdvProp.** [[paper](https://arxiv.org/abs/2204.09838)]
- [2021] **Learnable boundary guided adversarial training.** [[paper](http://openaccess.thecvf.com/content/ICCV2021/html/Cui_Learnable_Boundary_Guided_Adversarial_Training_ICCV_2021_paper.html)]
- [2020] **Smooth adversarial training.** [[paper](https://arxiv.org/abs/2006.14536)]
- [2020] **Adversarial examples improve image recognition.** [[paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Xie_Adversarial_Examples_Improve_Image_Recognition_CVPR_2020_paper.html)]
- [2020] **Attacks which do not kill training make adversarial learning stronger.** [[paper](https://proceedings.mlr.press/v119/zhang20z.html)]
- [2020] **Geometry-aware instance-reweighted adversarial training.** [[paper](https://arxiv.org/abs/2010.01736)]
- [2020] **A closer look at accuracy vs. robustness.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2020/hash/61d77652c97ef636343742fc3dcf3ba9-Abstract.html)]
- [2020] **Understanding and mitigating the tradeoff between robustness and accuracy.** [[paper](https://arxiv.org/abs/2002.10716)]
- [2019] **Interpolated adversarial training: Achieving robust neural networks without sacrificing too much accuracy.** [[paper](https://dl.acm.org/doi/abs/10.1145/3338501.3357369)]
- [2019] **Defense against adversarial attacks using feature scattering-based adversarial training.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2019/hash/d8700cbd38cc9f30cecb34f0c195b137-Abstract.html)]
- [2019] **Bilateral adversarial training: Towards fast training of more robust models against adversarial attacks.** [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Bilateral_Adversarial_Training_Towards_Fast_Training_of_More_Robust_Models_ICCV_2019_paper.html)]
- [2019] **Intriguing properties of adversarial training at scale.** [[paper](https://arxiv.org/abs/1906.03787)]

## 6 Comparison and Connection between Adversarial Training and Randomized Smoothing
Have been listed in  [Randomized Smoothing](../Randomized_Smoothing/README.md).

## 7 Defense against Patch Attack

- [2023] **Patchzero: Defending against adversarial patch attacks by detecting and zeroing the patch.** [[paper](https://openaccess.thecvf.com/content/WACV2023/html/Xu_PatchZero_Defending_Against_Adversarial_Patch_Attacks_by_Detecting_and_Zeroing_WACV_2023_paper.html)]
- [2019] **Local gradients smoothing: Defense against localized adversarial attacks.** [[paper](https://ieeexplore.ieee.org/abstract/document/8658401/)]

## 8 Multi-Attack Robustness


- [2023] **Towards compositional adversarial robustness: Generalizing adversarial training to composite semantic perturbations.** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Hsiung_Towards_Compositional_Adversarial_Robustness_Generalizing_Adversarial_Training_to_Composite_Semantic_CVPR_2023_paper.html)]
- [2022] **Formulating robustness against unforeseen attacks.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/392ac56724c133c37d5ea746e52f921f-Abstract-Conference.html)]
- [2021] **Perceptual adversarial robustness: Defense against unseen threat models.** [[paper](https://arxiv.org/abs/2006.12655)]
- [2020] **Confidence-calibrated adversarial training: Generalizing to unseen attacks.** [[paper](https://proceedings.mlr.press/v119/stutz20a.html)]
- [2020] **Adversarial self-supervised contrastive learning.** [[paper](https://proceedings.neurips.cc/paper/2020/hash/1f1baa5b8edac74eb4eaa329f14a0361-Abstract.html)]
- [2020] **Adversarial robustness against the union of multiple perturbation models.** [[paper](http://proceedings.mlr.press/v119/maini20a.html)]
- [2019] **Adversarial training and robustness for multiple perturbations.** [[paper](https://proceedings.neurips.cc/paper/2019/hash/5d4ae76f053f8f2516ad12961ef7fe97-Abstract.html)]
- [2019] **Adversarial framing for image and video classification.** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/5175)]

## 9 Cross-network and task Adversarial Training
- [2020] **A self-supervised approach for adversarial robustness.** [[paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Naseer_A_Self-supervised_Approach_for_Adversarial_Robustness_CVPR_2020_paper.html)]

## 10 Robust Pre-training and Fine-tuning

- [2024] **Securely fine-tuning pre-trained encoders against adversarial examples.** [[paper](https://ieeexplore.ieee.org/abstract/document/10646599/)]
- [2023] **Twins: A fine-tuning framework for improved transferability of adversarial robustness and generalization.** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Liu_TWINS_A_Fine-Tuning_Framework_for_Improved_Transferability_of_Adversarial_Robustness_CVPR_2023_paper.html)]
- [2023] **AutoLoRa: A parameter-free automated robust fine-tuning framework.** [[paper](https://arxiv.org/abs/2310.01818)]
- [2023] **Adversarial supervised contrastive learning.** [[paper](https://link.springer.com/article/10.1007/s10994-022-06269-7)]
- [2022] **Adversarial momentum-contrastive pre-training.** [[paper](https://www.sciencedirect.com/science/article/pii/S0167865522002161)]
- [2021] **When does contrastive learning preserve adversarial robustness from pretraining to fine-tuning?** [[paper](https://proceedings.neurips.cc/paper/2021/hash/b36ed8a07e3cd80ee37138524690eca1-Abstract.html)]
- [2020] **Adversarial robustness: From self-supervised pre-training to fine-tuning.** [[paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Adversarial_Robustness_From_Self-Supervised_Pre-Training_to_Fine-Tuning_CVPR_2020_paper.html)]
- [2020] **Robust pre-training by adversarial contrastive learning.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2020/hash/ba7e36c43aff315c00ec2b8625e3b719-Abstract.html)]
- [2020] **Adversarial self-supervised contrastive learning.** [[paper](https://proceedings.neurips.cc/paper/2020/hash/1f1baa5b8edac74eb4eaa329f14a0361-Abstract.html)]
- [2019] **Using pre-training can improve model robustness and uncertainty.** [[paper](https://proceedings.mlr.press/v97/hendrycks19a.html)]

## 11 Adaptive Perturbations


- [2024] **Improving adversarial training using vulnerability-aware perturbation budget.** [[paper](https://arxiv.org/abs/2403.04070)]
- [2023] **Improving robust fairness via balance adversarial training.** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26769)]
- [2023] **Cfa: Class-wise calibrated fair adversarial training.** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Wei_CFA_Class-Wise_Calibrated_Fair_Adversarial_Training_CVPR_2023_paper.html)]
- [2022] **CAT: Customized adversarial training for improved robustness.** [[paper](https://arxiv.org/abs/2002.06789)]
- [2021] **Understanding catastrophic overfitting in single-step adversarial training.** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16989)]
- [2021] **To be robust or to be fair: Towards fairness in adversarial training.** [[paper](http://proceedings.mlr.press/v139/xu21b.html)]
- [2019] **Instance adaptive adversarial training: Improved accuracy tradeoffs in neural nets.** [[paper](https://arxiv.org/abs/1910.08051)]
- [2018] **Mma training: Direct input space margin maximization through adversarial training.** [[paper](https://arxiv.org/abs/1812.02637)]

## 12 Efficiency

- [2024] **Improving fast adversarial training with prior-guided knowledge.** [[paper](https://ieeexplore.ieee.org/abstract/document/10478545/)]
- [2024] **Fast propagation is better: Accelerating single-step adversarial training via sampling subnetworks.** [[paper](https://ieeexplore.ieee.org/abstract/document/10471619/)]
- [2024] **Data filtering for efficient adversarial training.** [[paper](https://www.sciencedirect.com/science/article/pii/S0031320324001456)]
- [2023] **Adversarial coreset selection for efficient robust training.** [[paper]([https](https://link.springer.com/article/10.1007/s11263-023-01860-4))]
- [2023] **Efficient local linearity regularization to overcome catastrophic overfitting.** [[paper](https://arxiv.org/abs/2401.11618)]
- [2022] **Boosting fast adversarial training with learnable adversarial initialization.** [[paper](https://ieeexplore.ieee.org/abstract/document/9807638/)]
- [2022] **Prior-guided adversarial initialization for fast adversarial training.** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-19772-7_33)]
- [2021] **Bullettrain: Accelerating robust neural network training via boundary example mining.** [[paper](https://proceedings.neurips.cc/paper/2021/hash/9a1756fd0c741126d7bbd4b692ccbd91-Abstract.html)]
- [2019] **Fast is better than free: Revisiting adversarial training.** [[paper](https://arxiv.org/abs/2001.03994)]
- [2019] **You only propagate once: Accelerating adversarial training via maximal principle.** [[paper](https://proceedings.neurips.cc/paper/2019/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html)]
- [2019] **Adversarial training for free!** [[paper](https://proceedings.neurips.cc/paper/by-source-2019-1853)]

## 13 Adversarial Training for ViTs and Comparison with CNNs

- [2023] **A light recipe to train robust vision transformers.** [[paper](https://ieeexplore.ieee.org/abstract/document/10136149/)]
- [2022] **When adversarial training meets vision transformers: Recipes from training to architecture.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/760b5def8dcb1156aac454e9c0f5f406-Abstract-Conference.html)]
- [2022] **Towards efficient adversarial training on vision transformers.** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-19778-9_18)]
- [2021] **Are transformers more robust than cnns?** [[paper](https://proceedings.neurips.cc/paper/2021/hash/e19347e1c3ca0c0b97de5fb3b690855a-Abstract.html)]

## 14 Adversarial Training against Poisoning Attack
- [2023] **On the effectiveness of adversarial training against backdoor attacks.** [[paper](https://ieeexplore.ieee.org/abstract/document/10153093/)]

### 14.1 Adversarial Training against Backdoor Attack
- [2021] **Adversarial unlearning of backdoors via implicit hypergradient.** [[paper](https://arxiv.org/abs/2110.03735)]

### 14.2 Adversarial Training against Availability Attack

- [2023] **Learning the unlearnable: Adversarial augmentations suppress unlearnable example attacks.** [[paper](https://arxiv.org/abs/2303.15127)]
- [2021] **Better safe than sorry: Preventing delusive adversaries with adversarial training.** [[paper](https://proceedings.neurips.cc/paper/2021/hash/8726bb30dc7ce15023daa8ff8402bcfd-Abstract.html)]
- [2020] **Unlearnable examples: Making personal data unexploitable.** [[paper](https://arxiv.org/abs/2101.04898)]


