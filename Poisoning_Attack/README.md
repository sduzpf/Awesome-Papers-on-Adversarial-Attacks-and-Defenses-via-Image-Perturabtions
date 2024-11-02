# Poisoning Attack
- [1. Adversarial Perturbation-based Poisoning Attack](#1-Adversarial-Perturbation-based-Poisoning-Attack)
    - [1.1 Targeted Poisoning Attack](#11-Targeted-Poisoning-Attack)
    - [1.2 Backdoor (Trojan) Attack](#12-Backdoor-Trojan-Attack)
    - [1.3 Untargeted (Availability-Delusive-Indiscriminate) Attack](#13-Untargeted-Attack)
    - [1.4 Transferability](#34-Transferability)
      - [1.4.1 General Scenarios](#141-General-Scenarios)
      - [1.4.2 Downstream-agnostic Attack](#141-Downstream-agnostic-Attack)
    - [1.5 Imperceptibility](#15-Imperceptibility)
    - [1.6 Label-agnostic Attack](#16-Label-agnostic-Attack)
    - [1.7 Poisoning against Defense](#17-Poisoning-against-Defense)
    - [1.8 Connection between Evasion Attack and Poisoning Attack](#18-Connection-between-Evasion-Attack-and-Poisoning-Attack)
    - [1.9 Poisoning against Vision Transformer](#19-Poisoning-against-Vision-Transformer)
- [2. Non-adversarial Perturbation-based Poisoning Attack](#2-Non-adversarial-Perturbation-based-Poisoning-Attack)

   <!-- - [Citation](#citation) -->

##  1. Adversarial Perturbation-based Poisoning Attack

###  1.1 Targeted Poisoning Attack

- [2021] **Bullseye polytope: A scalable clean-label poisoning attack with improved transferability.** [[paper](https://ieeexplore.ieee.org/abstract/document/9581207/)]
- [2020] **Metapoison: Practical general-purpose clean-label data poisoning.** [[paper](https://proceedings.neurips.cc/paper/2020/hash/8ce6fc704072e351679ac97d4a985574-Abstract.html)]
- [2020] **Witches' brew: Industrial scale data poisoning via gradient matching.** [[paper](https://arxiv.org/abs/2009.02276)]
- [2019] **Transferable clean-label poisoning attacks on deep neural nets.** [[paper](http://proceedings.mlr.press/v97/zhu19a.html)]
- [2018] **Poison frogs! targeted clean-label poisoning attacks on neural networks.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2018/hash/22722a343513ed45f14905eb07621686-Abstract.html)]
- [2017] **Towards poisoning of deep learning algorithms with back-gradient optimization.** [[paper](https://dl.acm.org/doi/abs/10.1145/3128572.3140451)]
## 1.2 Backdoor (Trojan) Attack
- [2023] **Not all samples are born equal: Towards effective clean-label backdoor attacks.** [[paper](https://www.sciencedirect.com/science/article/pii/S0031320323002121)]
- [2023] **Narcissus: A practical clean-label backdoor attack with limited information.** [[paper](https://dl.acm.org/doi/abs/10.1145/3576915.3616617)]
- [2023] **An imperceptible data augmentation based blackbox clean-label backdoor attack on deep neural networks.** [[paper](https://ieeexplore.ieee.org/abstract/document/10208211/)]
  
- [2021] **Rethinking the backdoor attacks' triggers: A frequency perspective.** [[paper](http://openaccess.thecvf.com/content/ICCV2021/html/Zeng_Rethinking_the_Backdoor_Attacks_Triggers_A_Frequency_Perspective_ICCV_2021_paper.html)]
- [2021] **Lira: Learnable, imperceptible and robust backdoor attacks.** [[paper](http://openaccess.thecvf.com/content/ICCV2021/html/Doan_LIRA_Learnable_Imperceptible_and_Robust_Backdoor_Attacks_ICCV_2021_paper.html)]
- [2021] **Invisible backdoor attack with sample-specific triggers.** [[paper](http://openaccess.thecvf.com/content/ICCV2021/html/Li_Invisible_Backdoor_Attack_With_Sample-Specific_Triggers_ICCV_2021_paper.html)]
  
- [2020] **WaNet-imperceptible warping-based backdoor attack.** [[paper](https://arxiv.org/abs/2102.10369)]
- [2020] **Reflection backdoor: A natural backdoor attack on deep neural networks.** [[paper](https://link.springer.com/chapter/10.1007/978-3-030-58607-2_11)]
- [2020] **Backdoor embedding in convolutional neural network models via invisible perturbation.** [[paper](https://dl.acm.org/doi/abs/10.1145/3374664.3375751)]
- [2020] **Input-aware dynamic backdoor attack.** [[paper](https://proceedings.neurips.cc/paper/2020/hash/234e691320c0ad5b45ee3c96d0d7b8f8-Abstract.html)]
- [2020] **Invisible backdoor attacks on deep neural networks via steganography and regularization.** [[paper](https://ieeexplore.ieee.org/abstract/document/9186317/)]

- [2019] **A new backdoor attack in cnns by training set corruption without label poisoning.** [[paper](https://ieeexplore.ieee.org/abstract/document/8802997/)]
- [2019] **Label-consistent backdoor attacks.** [[paper](https://arxiv.org/abs/1912.02771)]
  
- [2018] **Trojaning attack on neural networks.** [[paper](https://scholarship.libraries.rutgers.edu/esploro/outputs/conferencePaper/Trojaning-attack-on-neural-networks/991031794682704646)]
- [2017] **Targeted backdoor attacks on deep learning systems using data poisoning.** [[paper](https://arxiv.org/abs/1712.05526)]
- [2017] **Badnets: Identifying vulnerabilities in the machine learning model supply chain.** [[paper](https://arxiv.org/abs/1708.06733)]

## 1.3 Untargeted (Availability-Delusive-Indiscriminate) Attack
- [2024] **Stable unlearnable example: Enhancing the robustness of unlearnable examples via stable error-minimizing noise.** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28169)]
- [2023] **Transferable unlearnable examples.** [[paper](https://arxiv.org/abs/2210.10114)]
- [2023] **Unlearnable clusters: Towards label-agnostic unlearnable examples.** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Unlearnable_Clusters_Towards_Label-Agnostic_Unlearnable_Examples_CVPR_2023_paper.html)]
- [2023] **Cuda: Convolution-based unlearnable datasets.** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Sadasivan_CUDA_Convolution-Based_Unlearnable_Datasets_CVPR_2023_paper.html)]
- [2022] **Availability attacks create shortcuts.** [[paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539241)]
- [2022] **Indiscriminate poisoning attacks on unsupervised contrastive learning.** [[paper](https://arxiv.org/abs/2202.11202)]
  
- [2021] **Neural tangent generalization attacks.** [[paper](https://proceedings.mlr.press/v139/yuan21b)]
- [2021] **Adversarial examples make strong poisons.** [[paper](https://proceedings.neurips.cc/paper/2021/hash/fe87435d12ef7642af67d9bc82a8b3cd-Abstract.html)]
- [2021] **Robust unlearnable examples: Protecting data privacy against adversarial learning.** [[paper](https://arxiv.org/abs/2203.14533)]
- [2020] **Unlearnable examples: Making personal data unexploitable.** [[paper](https://arxiv.org/abs/2101.04898)]
- [2019] **Learning to confuse: Generating training time adversarial data with auto-encoder.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2019/hash/1ce83e5d4135b07c0b82afffbe2b3436-Abstract.html)]
- [2018] **Neural tangent kernel: Convergence and generalization in neural networks.** [[paper](https://proceedings.neurips.cc/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html)]
- [2017] **Towards poisoning of deep learning algorithms with back-gradient optimization.** [[paper](https://dl.acm.org/doi/abs/10.1145/3128572.3140451)]
### 1.4 Transferability
#### 1.4.1 General Scenarios
- [2024] **Sharpness-aware data poisoning attack.** [[paper](https://arxiv.org/abs/2305.14851)]
- [2023] **Unlearnable clusters: Towards label-agnostic unlearnable examples.** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Unlearnable_Clusters_Towards_Label-Agnostic_Unlearnable_Examples_CVPR_2023_paper.html)]
- [2023] **Transferable unlearnable examples.** [[paper](https://arxiv.org/abs/2210.10114)]  
- [2021] **Just how toxic is data poisoning? a unified benchmark for backdoor and data poisoning attacks.** [[paper](http://proceedings.mlr.press/v139/schwarzschild21a.html)]
- [2021] **Bullseye polytope: A scalable clean-label poisoning attack with improved transferability.** [[paper](https://ieeexplore.ieee.org/abstract/document/9581207/)]
- [2020] **Metapoison: Practical general-purpose clean-label data poisoning.** [[paper](https://proceedings.neurips.cc/paper/2020/hash/8ce6fc704072e351679ac97d4a985574-Abstract.html)]
- [2020] **Witches' brew: Industrial scale data poisoning via gradient matching.** [[paper](https://arxiv.org/abs/2009.02276)]
- [2019] **Transferable clean-label poisoning attacks on deep neural nets.** [[paper](http://proceedings.mlr.press/v97/zhu19a.html)]
- [2019] **Why do adversarial attacks transfer? explaining transferability of evasion and poisoning attacks.** [[paper](https://www.usenix.org/conference/usenixsecurity19/presentation/demontis)]
  
#### 1.4.2 Downstream-agnostic Attack

##### Targeted poisoning
(https://arxiv.org/abs/1708.06733)]
- [2022] **PoisonedEncoder: Poisoning the Unlabeled Pre-training Data in Contrastive Learning.** [[paper](https://www.usenix.org/conference/usenixsecurity22/presentation/liu-hongbin)]

##### Backdoor attacks
- [2024] **Data poisoning based backdoor attacks to contrastive learning.** [[paper](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Data_Poisoning_based_Backdoor_Attacks_to_Contrastive_Learning_CVPR_2024_paper.html)]

- [2023] **An embarrassingly simple backdoor attack on self-supervised learning.** [[paper](http://openaccess.thecvf.com/content/ICCV2023/html/Li_An_Embarrassingly_Simple_Backdoor_Attack_on_Self-supervised_Learning_ICCV_2023_paper.html)]
- [2023] **Distribution preserving backdoor attack in self-supervised learning.** [[paper](https://ieeexplore.ieee.org/abstract/document/10646825/)]

- [2022] **Badencoder: Backdoor attacks to pre-trained encoders in self-supervised learning.** [[paper](https://ieeexplore.ieee.org/abstract/document/9833644/)]
- [2022] **Backdoor attacks on self-supervised learning.** [[paper](http://openaccess.thecvf.com/content/CVPR2022/html/Saha_Backdoor_Attacks_on_Self-Supervised_Learning_CVPR_2022_paper.html)]

##### Untargeted attacks
(https://arxiv.org/abs/1708.06733)]
- [2022] **Indiscriminate poisoning attacks on unsupervised contrastive learning.** [[paper](https://arxiv.org/abs/2202.11202)]
  
### 1.5 Imperceptibility
- [2024] **A dual stealthy backdoor: From both spatial and frequency perspectives.** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/27954)]
- [2024] **SPY-watermark: Robust invisible watermarking for backdoor attack.** [[paper](https://ieeexplore.ieee.org/abstract/document/10448363/)]
- [2023] **Clean-label poisoning attack with perturbation causing dominant features.** [[paper](https://www.sciencedirect.com/science/article/pii/S0020025523004474)]
- [2022] **Sleeper agent: Scalable hidden trigger backdoors for neural networks trained from scratch.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/79eec295a3cd5785e18c61383e7c996b-Abstract-Conference.html)]
- [2022] **Dynamic backdoor attacks against machine learning models.** [[paper](https://ieeexplore.ieee.org/abstract/document/9797338/)]
- [2021] **Deep feature space trojan attack of neural networks by controlled detoxification.** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16201)]
- [2021] **Rethinking the backdoor attacks' triggers: A frequency perspective.** [[paper](http://openaccess.thecvf.com/content/ICCV2021/html/Zeng_Rethinking_the_Backdoor_Attacks_Triggers_A_Frequency_Perspective_ICCV_2021_paper.html)]
- [2021] **Bullseye polytope: A scalable clean-label poisoning attack with improved transferability.** [[paper](https://ieeexplore.ieee.org/abstract/document/9581207/)]
- [2020] **Invisible backdoor attacks on deep neural networks via steganography and regularization.** [[paper](https://ieeexplore.ieee.org/abstract/document/9186317/)]
- [2020] **Input-aware dynamic backdoor attack.** [[paper](https://proceedings.neurips.cc/paper/2020/hash/234e691320c0ad5b45ee3c96d0d7b8f8-Abstract.html)]
- [2020] **Backdooring and poisoning neural networks with image-scaling attacks.** [[paper](https://ieeexplore.ieee.org/abstract/document/9283824/)]
- [2020] **Reflection backdoor: A natural backdoor attack on deep neural networks.** [[paper](https://link.springer.com/chapter/10.1007/978-3-030-58607-2_11)]
- [2020] **Backdoor embedding in convolutional neural network models via invisible perturbation.** [[paper](https://dl.acm.org/doi/abs/10.1145/3374664.3375751)]
- [2020] **Hidden trigger backdoor attacks.** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6871)]

- [2019] **Transferable clean-label poisoning attacks on deep neural nets.** [[paper](http://proceedings.mlr.press/v97/zhu19a.html)]
- [2019] **A new backdoor attack in cnns by training set corruption without label poisoning.** [[paper](https://ieeexplore.ieee.org/abstract/document/8802997/)]

- [2018] **Poison frogs! targeted clean-label poisoning attacks on neural networks.** [[paper](https://proceedings.neurips.cc/paper_files/paper/2018/hash/22722a343513ed45f14905eb07621686-Abstract.html)]
### 1.6 Label-agnostic Attack

- [2023] **Unlearnable clusters: Towards label-agnostic unlearnable examples.** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Unlearnable_Clusters_Towards_Label-Agnostic_Unlearnable_Examples_CVPR_2023_paper.html)]

### 1.7 Poisoning against Defense
- [2024] **Stable unlearnable example: Enhancing the robustness of unlearnable examples via stable error-minimizing noise.** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28169)]
- [2024] **Re-thinking data availability attacks against deep neural networks.** [[paper](https://openaccess.thecvf.com/content/CVPR2024/html/Fang_Re-thinking_Data_Availability_Attacks_Against_Deep_Neural_Networks_CVPR_2024_paper.html)]
- [2024] **SPY-watermark: Robust invisible watermarking for backdoor attack.** [[paper](https://ieeexplore.ieee.org/abstract/document/10448363/)] 
- [2021] **How robust are randomized smoothing based defenses to data poisoning?** [[paper](http://openaccess.thecvf.com/content/CVPR2021/html/Mehra_How_Robust_Are_Randomized_Smoothing_Based_Defenses_to_Data_Poisoning_CVPR_2021_paper.html)]
- [2021] **Robust unlearnable examples: Protecting data privacy against adversarial learning.** [[paper](https://arxiv.org/abs/2203.14533)]

- [2020] **Input-aware dynamic backdoor attack.** [[paper](https://proceedings.neurips.cc/paper/2020/hash/234e691320c0ad5b45ee3c96d0d7b8f8-Abstract.html)]
- [2020] **WaNet-imperceptible warping-based backdoor attack.** [[paper](https://arxiv.org/abs/2102.10369)]

### 1.8 Connection between Evasion Attack and Poisoning Attack
- [2022] **Can adversarial training be manipulated by non-robust features?** [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a94a8800a4b0af45600bab91164849df-Abstract-Conference.html)]
- [2021] **Adversarial examples make strong poisons.** [[paper](https://proceedings.neurips.cc/paper/2021/hash/fe87435d12ef7642af67d9bc82a8b3cd-Abstract.html)]

### 1.9 Poisoning against Vision Transformer
- [2024] **A closer look at robustness of vision transformers to backdoor attacks.** [[paper](https://openaccess.thecvf.com/content/WACV2024/html/Subramanya_A_Closer_Look_at_Robustness_of_Vision_Transformers_to_Backdoor_WACV_2024_paper.html)]
- [2023] **Defending backdoor attacks on vision transformer via patch processing.** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25125)]
- [2023] **Trojvit: Trojan insertion in vision transformers.** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Zheng_TrojViT_Trojan_Insertion_in_Vision_Transformers_CVPR_2023_paper.html)]

# 2. Adversarial Perturbation-based Poisoning Attack
- [2021] **Adversarial examples make strong poisons.** [[paper](https://proceedings.neurips.cc/paper/2021/hash/fe87435d12ef7642af67d9bc82a8b3cd-Abstract.html)]
- [2023] **Batt: Backdoor attack with transformation-based triggers.** [[paper](https://ieeexplore.ieee.org/abstract/document/10096034/)]
