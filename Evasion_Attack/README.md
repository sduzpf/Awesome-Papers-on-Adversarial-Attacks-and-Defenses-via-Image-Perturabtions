# EVASION ATTACK

If not explicitly indicated, works are proposed for CNNs. Attacks for other networks or models would be explicitly listed.

  - [1 Non-adversarial Perturbation-based Evasion Attack](#1-Non-adversarial-Perturbation-based-Evasion-Attack)
    - [1.1 Current Studies and Underlying Reasons of Vulnerability for CNNs](#11-Current-Studies-and-Underlying-Reasons-of-Vulnerability-for-CNNs)
    - [1.2 Attack against Vision Transformers](#12-Attack-against-Vision-Transformers)
    - [1.3 Comparison between CNNs and ViTs](#13-Comparison-between-CNNs-and-ViTs)
  - [2 Non-adversarial Perturbation-based Poisoning Attack](#2-Non-adversarial-Perturbation-based-Poisoning-Attack)
  - [3 Adversarial Perturbation-based Evasion Attack](#3-Adversarial-Perturbation-based-Evasion-Attack)
    - [3.1 Basic Generation Methods](#31-Basic-Generation-Methods)
    - [3.2 Underlying Reasons of Vulnerability](#32-Underlying-Reasons-of-Vulnerability)
    - [3.3 Black-box Attack](#33-Black-box-Attack)
      - [3.3.1 Query-based Attack](#331-Query-based-Attack)
      - [3.3.2 Query Efficiency](#332-Query-Efficiency)
      - [3.3.3 Transfer-based Attack and Adversarial Transferability (Underlying Reasons)](#333-transfer-based-attack-and-adversarial-transferability-underlying-reasons)
      - [3.3.4 Adversarial Transferability Enhancement](#334-Adversarial-Transferability-Enhancement)
        - [3.3.4.1 Data augmentation](#3341-Data-augmentation)
        - [3.3.4.2 Ensemble-based techniques](#3342-Ensemble-based-techniques)
        - [3.3.4.3 Momentum-based methods](#3343-Momentum-based-methods)
        - [3.3.4.4 Architecture-oriented methods](#3344-Architecture-oriented-methods)
        - [3.3.4.5 Finding proper substitute models](#3345-Finding-proper-substitute-models)
        - [3.3.4.6 Distribution-oriented methods](#3346-Distribution-oriented-methods)
        - [3.3.4.7 Other methods](#3347-Other-methods)
      - [3.3.5 Cross-domain and modality Transferability](#335-Cross-domain-and-modality-Transferability)
      - [3.3.6 Cross-task Transferability](#336-Cross-task-Transferability)
    - [3.4 Perturbation against Vision Transformer and Cross-architecture Transferability](#34-Perturbation-against-Vision-Transformer-and-Cross-architecture-Transferability)
      - [3.4.1 Current studies and Underlying Reasons for Vulnerability](#341-Current-studies-and-Underlying-Reasons-for-Vulnerability)
      - [3.4.2 Comparison and Cross-architecture Transferability between CNNs and ViTs](#342-Comparison-and-Cross-architecture-Transferability-between-CNNs-and-ViTs)
    - [3.5 Non-box Attack](#35-Non-box-Attack)
    - [3.6 Attack against Defense](#36-Attack-against-Defense)
    - [3.7 Problem of the Cross-entropy Loss](#37-Problem-of-the-Cross-entropy-Loss)
    - [3.8 Imperceptibility](#38-Imperceptibility)
    - [3.9 Diverse Perturbations](#39-Diverse-Perturbations)
      - [3.9.1 Beyond l_p-norm Perturbations](#2391-Beyond-l_p-norm-Perturbations)
      - [3.9.2 Beyond Single or Dual-type Perturbations](#2392-Beyond-Single-or-Dual-type-Perturbations)
    - [3.10 Unconstrained Perturbations](#310-Unconstrained-Perturbations)
  - [4. Adversarial Perturbation-based Poisoning Attack](#4-Adversarial-Perturbation-based-Poisoning-Attack)
    - [4.1 Targeted Poisoning Attack](#41-Targeted-Poisoning-Attack)
    - [4.2 Backdoor (Trojan) Attack](#42-Backdoor-Trojan-Attack)
    - [4.3 Untargeted (Availability) Attack](#43-Untargeted-Availability-Attack)
    - [4.4 Transferability](#44-Transferability)
      - [4.4.1 Downstream-agnostic Attack](#441-Downstream-agnostic-Attack)
        - [4.4.1.1 Targeted Poisoning](#4411-Targeted-poisoning)
        - [4.4.1.2 Backdoor attacks](#4412-Backdoor-attacks)
        - [4.4.1.3 Untargeted attacks](#4413-Untargeted-attacks)
      - [4.5 Imperceptibility](#45-Imperceptibility)
      - [4.6 Label-agnostic Attack](#46-Label-agnostic-Attack)
      - [4.7 Poisoning against Defense](#47-Poisoning-against-Defense)
      - [4.8 Connection between Evasion Attack and Poisoning Attack](#48-Connection-between-Evasion-Attack-and-Poisoning-Attack)
      - [4.9 Poisoning against Vision Transformer](#49-Poisoning-against-Vision-Transformer)
      - [4.10 Efficiency](#410-Efficiency)
  <!-- - [Citation](#citation) -->

# PAPER LIST

## 1 Non-adversarial Perturbation-based Evasion Attack

### 1.1 Current Studies and Underlying Reasons of Vulnerability for CNNs

- [2023] **Delving deeper into anti-aliasing in convnets** [[paper](https://link.springer.com/article/10.1007/s11263-022-01672-y)]
  
- [2023] **Anti-aliasing deep image classifiers using novel depth adaptive blurring and activation function** [[paper](https://www.sciencedirect.com/science/article/pii/S0925231223002473)]
  
- [2023] **Alias-free convnets: Fractional shift invariance via polynomial activations** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Michaeli_Alias-Free_Convnets_Fractional_Shift_Invariance_via_Polynomial_Activations_CVPR_2023_paper.html)]
  
- [2022] **Learnable polyphase sampling for shift invariant and equivariant convolutional networks** [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e87b1e06be8c3594c810e8991e77ea40-Abstract-Conference.html)]

- [2021] **Batch normalization increases adversarial vulnerability and decreases adversarial transferability: A non-robust feature perspective** [[paper](https://proceedings.neurips.cc/paper/2019/hash/3eefceb8087e964f89c2d59e8a249915-Abstract.html)]
  
- [2021] **The many faces of robustness: A critical analysis of out-of-distribution generalization** [[paper](http://openaccess.thecvf.com/content/ICCV2021/html/Hendrycks_The_Many_Faces_of_Robustness_A_Critical_Analysis_of_Out-of-Distribution_ICCV_2021_paper.html)]

- [2021] **Truly shift-invariant convolutional neural networks** [[paper](http://openaccess.thecvf.com/content/CVPR2021/html/Chaman_Truly_Shift-Invariant_Convolutional_Neural_Networks_CVPR_2021_paper.html)]

- [2019] **Learning robust global representations by penalizing local predictive power** [[paper](https://proceedings.neurips.cc/paper/2019/hash/3eefceb8087e964f89c2d59e8a249915-Abstract.html)]

- [2019] **Making convolutional networks shift-invariant again** [[paper](http://proceedings.mlr.press/v97/zhang19a.html)]
  
- [2019] **Why do deep convolutional networks generalize so poorly to small image transformations?** [[paper](https://www.jmlr.org/papers/v20/19-519.html)]

- [2019] **Exploring the landscape of spatial robustness** [[paper]([https://arxiv.org/abs/1706.06969](http://proceedings.mlr.press/v97/engstrom19a.html?utm_medium=email&utm_source=transaction))]
  
- [2018] **Benchmarking neural network robustness to common corruptions and perturbations** [[paper](https://arxiv.org/abs/1903.12261)]

- [2018] **A rotation and a translation suffice: Fooling cnns with simple transformations** [[paper](https://openreview.net/forum?id=BJfvknCqFQ)]

- [2018] **ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness** [[paper](https://arxiv.org/abs/1811.12231)]
  
- [2018] **Semantic adversarial examples** [[paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w32/html/Hosseini_Semantic_Adversarial_Examples_CVPR_2018_paper.html)]

- [2017] **A study and comparison of human and deep learning recognition performance under visual distortions** [[paper](https://arxiv.org/abs/1705.02498)]

- [2017] **Comparing deep neural networks against humans: Object recognition when the signal gets weaker** [[paper](https://arxiv.org/abs/1706.06969)]
  
- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

### 1.2 Attack against Vision Transformers

- [2024] **Making vision transformers truly shift-equivariant** [[paper](https://openaccess.thecvf.com/content/CVPR2024/html/Rojas-Gomez_Making_Vision_Transformers_Truly_Shift-Equivariant_CVPR_2024_paper.html)]
  
- [2023] **Classification robustness to common optical aberrations** [[paper](https://openaccess.thecvf.com/content/ICCV2023W/AROW/html/Muller_Classification_Robustness_to_Common_Optical_Aberrations_ICCVW_2023_paper.html)]

- [2023] **Improving robustness of vision transformers by reducing sensitivity to patch corruptions** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Guo_Improving_Robustness_of_Vision_Transformers_by_Reducing_Sensitivity_To_Patch_CVPR_2023_paper.html)]

- [2022] **Are vision transformers robust to patch perturbations?** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-19775-8_24)]
  
- [2022] **Assaying out-of-distribution generalization in transfer learning** [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2f5acc925919209370a3af4eac5cad4a-Abstract-Conference.html)]

- [2022] **Towards robust vision transformer** [[paper](http://openaccess.thecvf.com/content/CVPR2022/html/Mao_Towards_Robust_Vision_Transformer_CVPR_2022_paper.html)]

- [2022] **A comprehensive study of image classification model sensitivity to foregrounds, backgrounds, and visual attributes** [[paper](http://openaccess.thecvf.com/content/CVPR2022/html/Moayeri_A_Comprehensive_Study_of_Image_Classification_Model_Sensitivity_to_Foregrounds_CVPR_2022_paper.html)]

- [2022] **Understanding and improving robustness of vision transformers through patch-based negative augmentation** [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67662aa16456e0df65ab001136f92fd0-Abstract-Conference.html)]

- [2021] **Blending anti-aliasing into vision transformer** [[paper](https://proceedings.neurips.cc/paper/2021/hash/2b3bf3eee2475e03885a110e9acaab61-Abstract.html)]

- [2021] **Understanding robustness of transformers for image classification** [[paper](http://openaccess.thecvf.com/content/ICCV2021/html/Bhojanapalli_Understanding_Robustness_of_Transformers_for_Image_Classification_ICCV_2021_paper.html)]

- [2021] **Intriguing properties of vision transformers** [[paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/c404a5adbf90e09631678b13b05d9d7a-Abstract.html)]
  
- [2019] **Exploring the landscape of spatial robustness** [[paper](http://proceedings.mlr.press/v97/engstrom19a.html)]
  
### 1.3 Comparison between CNNs and ViTs
- [2022] **A comprehensive study of image classification model sensitivity to foregrounds, backgrounds, and visual attributes** [[paper](http://openaccess.thecvf.com/content/CVPR2022/html/Moayeri_A_Comprehensive_Study_of_Image_Classification_Model_Sensitivity_to_Foregrounds_CVPR_2022_paper.html)]
  
- [2022] **Are vision transformers robust to patch perturbations?** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-19775-8_24)]
  
- [2021] **Intriguing properties of vision transformers** [[paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/c404a5adbf90e09631678b13b05d9d7a-Abstract.html)]

- [2022] **Towards robust vision transformer** [[paper](http://openaccess.thecvf.com/content/CVPR2022/html/Mao_Towards_Robust_Vision_Transformer_CVPR_2022_paper.html)]

- [2021] **Understanding robustness of transformers for image classification** [[paper](http://openaccess.thecvf.com/content/ICCV2021/html/Bhojanapalli_Understanding_Robustness_of_Transformers_for_Image_Classification_ICCV_2021_paper.html)]

- [2019] **Exploring the landscape of spatial robustness** [[paper](http://proceedings.mlr.press/v97/engstrom19a.html)]
  
## 2 Non-adversarial Perturbation-based Poisoning Attack

### Availability attacks
- [2021] **Adversarial examples make strong poisons** [[paper](https://proceedings.neurips.cc/paper/2021/hash/fe87435d12ef7642af67d9bc82a8b3cd-Abstract.html)]

### Backdoor attacks
- [2023] **Batt: Backdoor attack with transformation-based triggers** [[paper](https://ieeexplore.ieee.org/abstract/document/10096034/)]

### 3 Adversarial Perturbation-based Evasion Attack

#### 3.1 Basic Generation Methods
- [2013] [L-BFGS] **Intriguing properties of neural networks** [[paper](https://gitea.sharpe6.com/Adog64/Adversarial-Machine-Learning-Clinic/raw/commit/0e94757c07259c82060fd1d4626c5c8d471deb43/references/Intriguing_properties_of_neural_networks.pdf)]
  
- [2014] [FGSM] **Adversarial examples in the physical world** [[paper](https://arxiv.org/abs/1412.6572)]
  
- [2016] [JSMA] **The limitations of deep learning in adversarial settings** [[paper](https://ieeexplore.ieee.org/abstract/document/7467366/)]
    
- [2016] [DeepFool] **Deepfool: A simple and accurate method to fool deep neural networks** [[paper](http://openaccess.thecvf.com/content_cvpr_2016/html/Moosavi-Dezfooli_DeepFool_A_Simple_CVPR_2016_paper.html)]

- [2016] [I-FGSM] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://www.taylorfrancis.com/chapters/edit/10.1201/9781351251389-8/adversarial-examples-physical-world-alexey-kurakin-ian-goodfellow-samy-bengio)]
  
- [2017] [C&W] **Towards evaluating the robustness of neural networks** [[paper](https://ieeexplore.ieee.org/abstract/document/7958570/)]
  
- [2017] [PGD] **Towards deep learning models resistant to adversarial attacks** [[paper](https://www.utdallas.edu/~mxk055100/courses/adv-ml-19f/1706.06083.pdf)]
  
- [2018] [MI-FGSM] **Boosting adversarial attacks with momentum** [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.html)]

##### Generative methods

- [2018] [Residual-like networks] **Learning to attack: Adversarial transformation networks** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/11672)]
  
- [2018] [Encoder-decoder networks] **Generative adversarial perturbations** [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Poursaeed_Generative_Adversarial_Perturbations_CVPR_2018_paper.html)]

- [2019] [Encoder-decoder networks] **Black-box adversarial attack with transferable model-based embedding** [[paper](https://arxiv.org/abs/1911.07140)]
  
- [2019] [Encoder-decoder networks] **Autozoom: Autoencoder-based zeroth order optimization method for attacking black-box neural networks** [[paper](https://aaai.org/ojs/index.php/AAAI/article/view/3852)]

- [2018] [GANs] **Generating adversarial examples with adversarial networks** [[paper](https://arxiv.org/abs/1801.02610)]

##### Search-based methods
- [2020] [Random search] **Colorfool: Semantic adversarial colorization** [[paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Shamsabadi_ColorFool_Semantic_Adversarial_Colorization_CVPR_2020_paper.html)]
  
- [2018] [Random search] **Semantic adversarial examples** [[paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w32/html/Hosseini_Semantic_Adversarial_Examples_CVPR_2018_paper.html)]

- [2023] [Exhaustive search] **Alias-free convnets: Fractional shift invariance via polynomial activations** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Michaeli_Alias-Free_Convnets_Fractional_Shift_Invariance_via_Polynomial_Activations_CVPR_2023_paper.html)]

- [2022] [Exhaustive search] **Natural color fool: Towards boosting black-box unrestricted attacks** [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/31d0d59fe946684bb228e9c8e887e176-Abstract-Conference.html)]
  
- [2019] [Exhaustive search] **Exploring the landscape of spatial robustness** [[paper](http://proceedings.mlr.press/v97/engstrom19a.html)]

- [2019] [Exhaustive search] **Exploring the landscape of spatial robustness** [[paper](http://proceedings.mlr.press/v97/engstrom19a.html)]

- [2020] [Population-based methods] **Adv-watermark: A novel watermark perturbation for adversarial examples** [[paper](https://dl.acm.org/doi/abs/10.1145/3394171.3413976)]
  
- [2019] [Population-based methods] **One pixel attack for fooling deep neural networks** [[paper](https://ieeexplore.ieee.org/abstract/document/8601309/)]

#### 3.2 Underlying Reasons of Vulnerability
- [2021] **Batch normalization increases adversarial vulnerability and decreases adversarial transferability: A non-robust feature perspective** [[paper](http://openaccess.thecvf.com/content/ICCV2021/html/Benz_Batch_Normalization_Increases_Adversarial_Vulnerability_and_Decreases_Adversarial_Transferability_A_ICCV_2021_paper.html)]
- 
- [2021] **Natural adversarial examples** [[paper](http://openaccess.thecvf.com/content/CVPR2021/html/Hendrycks_Natural_Adversarial_Examples_CVPR_2021_paper.html)]

- [2021] **Adversarial robustness through the lens of causality** [[paper](https://arxiv.org/abs/2106.06196)]
 
- [2020] **A causal view on robustness of neural networks** [[paper](https://proceedings.neurips.cc/paper_files/paper/2020/hash/02ed812220b0705fabb868ddbf17ea20-Abstract.html)]
   
- [2019] **Adversarial examples are not bugs, they are features** [[paper](https://proceedings.neurips.cc/paper/2019/hash/e2c420d928d4bf8ce0ff2ec19b371514-Abstract.html)]

- [2018] **ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness** [[paper](https://arxiv.org/abs/1811.12231)]


#### 3.3 Black-box Attack
##### 3.3.1 Query-based Attack
###### Gradient estimation based attacks

- [2020] **Towards query-efficient black-box adversary with zeroth-order natural gradient descent** [[paper](https://aaai.org/ojs/index.php/AAAI/article/view/6173)]

- [2019] **Autozoom: Autoencoder-based zeroth order optimization method for attacking black-box neural networks** [[paper](https://aaai.org/ojs/index.php/AAAI/article/view/3852)]
  
- [2019] **Prior convictions: Black-box adversarial attacks with bandits and priors** [[paper](https://arxiv.org/abs/1807.07978)]

- [2019] **Black-box adversarial attack with transferable model-based embedding** [[paper](https://arxiv.org/abs/1911.07140)]

- [2019] **Nattack: Learning the distributions of adversarial examples for an improved black-box attack on deep neural networks** [[paper]([https://arxiv.org/abs/1611.05760](http://proceedings.mlr.press/v97/li19g.html?ref=https://githubhelp.com))]

- [2018] **Black-box adversarial attacks with limited queries and information** [[paper](https://proceedings.mlr.press/v80/ilyas18a.html)]
  
- [2018] **Practical black-box attacks on deep neural networks using efficient query mechanisms** [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Arjun_Nitin_Bhagoji_Practical_Black-box_Attacks_ECCV_2018_paper.html)]

- [2017] **Zoo: Zeroth order optimization based black-box attacks to deep neural networks without training substitute models** [[paper](https://dl.acm.org/doi/abs/10.1145/3128572.3140448)]
  
###### Gradient estimation based attacks
  
- [2020] **HopSkipJumpAttack: A query-efficient decision-based attack** [[paper](https://ieeexplore.ieee.org/abstract/document/9152788/)]

###### Combination of query-based and transfer-based attacks
- [2022] **Blackbox attacks via surrogate ensemble search** [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/23b9d4e18b151ba2108fb3f1efaf8de4-Abstract-Conference.html)]
  
- [2021] **Query-efficient black-box adversarial attacks guided by a transfer-based prior** [[paper](https://ieeexplore.ieee.org/abstract/document/9609659/)]

- [2021] **Simulating unknown target models for query-efficient black-box attacks** [[paper](http://openaccess.thecvf.com/content/CVPR2021/html/Ma_Simulating_Unknown_Target_Models_for_Query-Efficient_Black-Box_Attacks_CVPR_2021_paper.html)]
- [2019] **Improving black-box adversarial attacks with a transfer-based prior** [[paper](https://proceedings.neurips.cc/paper/2019/hash/32508f53f24c46f685870a075eaaa29c-Abstract.html)]

###### Search-based attacks 
- [2023] **Query-efficient decision-based black-box patch attack** [[paper](https://ieeexplore.ieee.org/abstract/document/10227335/)]

- [2022] **Sparse-RS: a versatile framework for query-efficient sparse black-box adversarial attacks** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20595)]
  
- [2020] **Square attack: A query-efficient black-box adversarial attack via random search** [[paper](https://link.springer.com/chapter/10.1007/978-3-030-58592-1_29)]

###### Sampling-based attacks 

- [2019] **Guessing smart: Biased sampling for efficient black-box adversarial attacks** [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Brunner_Guessing_Smart_Biased_Sampling_for_Efficient_Black-Box_Adversarial_Attacks_ICCV_2019_paper.html)]

###### Geometric-based attacks

- [2022] **Triangle attack: A query-efficient decision-based adversarial attack** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-20065-6_10)]

  
##### 3.3.2 Query Efficiency

- [2023] **Decision-based query efficient adversarial attack via adaptive boundary learning** [[paper](https://ieeexplore.ieee.org/abstract/document/10163476/)]

- [2023] **Robustness with query-efficient adversarial attack using reinforcement learning** [[paper](https://openaccess.thecvf.com/content/CVPR2023W/AML/html/Sarkar_Robustness_With_Query-Efficient_Adversarial_Attack_Using_Reinforcement_Learning_CVPRW_2023_paper.html)]

- [2023] **Query-efficient decision-based black-box patch attack** [[paper](https://ieeexplore.ieee.org/abstract/document/10227335/)]
  
- [2022] **Triangle attack: A query-efficient decision-based adversarial attack** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-20065-6_10)]

- [2022] **Blackbox attacks via surrogate ensemble search** [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/23b9d4e18b151ba2108fb3f1efaf8de4-Abstract-Conference.html)]

- [2021] **Simulating unknown target models for query-efficient black-box attacks** [[paper](http://openaccess.thecvf.com/content/CVPR2021/html/Ma_Simulating_Unknown_Target_Models_for_Query-Efficient_Black-Box_Attacks_CVPR_2021_paper.html)]

- [2021] **Query-efficient black-box adversarial attacks guided by a transfer-based prior** [[paper](https://ieeexplore.ieee.org/abstract/document/9609659/)]

- [2020] **Improving query efficiency of black-box adversarial attack** [[paper](https://ui.adsabs.harvard.edu/abs/2020arXiv200911508B/abstract)]

- [2020] **Towards query-efficient black-box adversary with zeroth-order natural gradient descent** [[paper](https://ui.adsabs.harvard.edu/abs/2020arXiv200207891Z/abstract)]
- [2020] **HopSkipJumpAttack: A query-efficient decision-based attack** [[paper](https://ieeexplore.ieee.org/abstract/document/9152788/)]

- [2020] **Qeba: Query-efficient boundary-based blackbox attack** [[paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Li_QEBA_Query-Efficient_Boundary-Based_Blackbox_Attack_CVPR_2020_paper.html)]
  
- [2020] **Square attack: A query-efficient black-box adversarial attack via random search** [[paper](https://link.springer.com/chapter/10.1007/978-3-030-58592-1_29)]

- [2019] **Guessing smart: Biased sampling for efficient black-box adversarial attacks** [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Brunner_Guessing_Smart_Biased_Sampling_for_Efficient_Black-Box_Adversarial_Attacks_ICCV_2019_paper.html)]
  
- [2019] **Improving black-box adversarial attacks with a transfer-based prior** [[paper](https://proceedings.neurips.cc/paper/2019/hash/32508f53f24c46f685870a075eaaa29c-Abstract.html)]

- [2019] **[Prior convictions: Black-box adversarial attacks with bandits and priors]** [[paper](https://arxiv.org/abs/1807.07978)]

- [2019] **Black-box adversarial attack with transferable model-based embedding** [[paper](https://arxiv.org/abs/1911.07140)]

- [2019] **Nattack: Learning the distributions of adversarial examples for an improved black-box attack on deep neural networks** [[paper](http://proceedings.mlr.press/v97/li19g.html?ref=https://githubhelp.com)]

- [2019] **Autozoom: Autoencoder-based zeroth order optimization method for attacking black-box neural networks** [[paper](https://aaai.org/ojs/index.php/AAAI/article/view/3852)]
  
- [2018] **Black-box adversarial attacks with limited queries and information** [[paper](https://proceedings.mlr.press/v80/ilyas18a.html)]
  
- [2018] **Practical black-box attacks on deep neural networks using efficient query mechanisms** [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Arjun_Nitin_Bhagoji_Practical_Black-box_Attacks_ECCV_2018_paper.html)]
  
##### 3.3.3 Transfer-based Attack and Adversarial Transferability (Underlying Reasons)

- [2023] **Why does little robustness help? a further step towards understanding adversarial transferability** [[paper](https://ieeexplore.ieee.org/abstract/document/10646840/)]
- [2022] **Toward understanding and boosting adversarial transferability from a distribution perspective** [[paper](https://ieeexplore.ieee.org/abstract/document/9917370/)]
  
- [2021] **Trs: Transferability reduced ensemble via promoting gradient diversity and model smoothness** [[paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/937936029af671cf479fa893db91cbdd-Abstract.html)]
  
- [2020] **Towards understanding and improving the transferability of adversarial examples in deep neural networks** [[paper](https://proceedings.mlr.press/v129/wu20a.html)]

- [2019] **Guessing smart: Biased sampling for efficient black-box adversarial attacks** [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Brunner_Guessing_Smart_Biased_Sampling_for_Efficient_Black-Box_Adversarial_Attacks_ICCV_2019_paper.html)]

- [2019] **Skip connections matter: On the transferability of adversarial examples generated with ResNets** [[paper](https://arxiv.org/abs/2002.05990)]

- [2019] **Adversarial examples are not bugs, they are features** [[paper](https://proceedings.neurips.cc/paper/2019/hash/e2c420d928d4bf8ce0ff2ec19b371514-Abstract.html)]

- [2019] **Why do adversarial attacks transfer? explaining transferability of evasion and poisoning attacks** [[paper](https://www.usenix.org/conference/usenixsecurity19/presentation/demontis)]

- [2018] **Is robustness the cost of accuracy?--a comprehensive study on the robustness of 18 deep image classification models** [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Dong_Su_Is_Robustness_the_ECCV_2018_paper.html)]

- [2017] **The space of transferable adversarial examples** [[paper](https://arxiv.org/abs/1704.03453)]

- [2016] **Delving into transferable adversarial examples and black-box attacks** [[paper](https://arxiv.org/abs/1611.02770)]
- [2014] **Explaining and harnessing adversarial examples** [[paper](https://arxiv.org/abs/1412.6572)]
##### 3.3.4 Adversarial Transferability Enhancement
###### 3.3.4.1 Data augmentation
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]
- 
###### 3.3.4.2 Ensemble-based techniques
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]
###### 3.3.4.3 Momentum-based methods
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]
###### 3.3.4.4 Architecture-oriented methods
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]
###### 3.3.4.5 Finding proper substitute models
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]
###### 3.3.4.6 Distribution-oriented methods
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]
###### 3.3.4.7 Other methods
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

##### 3.3.5 Cross-domain and modality Transferability
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]
- 
##### 3.3.6 Cross-task Transferability
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

#### 3.4 Perturbation against Vision Transformer and Cross-architecture Transferability
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

##### 3.4.1 Current Studies and Underlying Reasons for Vulnerability
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

##### 3.4.2 Comparison and Cross-architecture Transferability between CNNs and ViTs
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

#### 3.5 Non-box Attack
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

#### 3.6 Attack against Defense
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

#### 3.7 Problem of the Cross-entropy Loss
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

#### 3.8 Imperceptibility
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

#### 3.9 Diverse Perturbations
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

##### 3.9.1 Beyond l_p-norm Perturbations
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]
##### 3.9.2 Beyond Single or Dual-type Perturbations
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

#### 3.10 Unconstrained Perturbations
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

## 4. Adversarial Perturbation-based Poisoning Attack

### 4.1 Targeted Poisoning Attack

- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

### 4.2 Backdoor (Trojan) Attack

#### 4.2.1 Alleviating Hallucination of LLMs

##### survey

- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

### 4.3 Untargeted (Availability) Attack
- Untargeted (Availability, Delusive, Indiscriminate) Attack

- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]
### 4.4 Transferability
#### 4.4.1 Targeted poisoning

##### 4.4.1.1 Targeted poisoning
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

##### 4.4.1.2 Backdoor attacks
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

##### 4.4.1.3 Untargeted attacks
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

### 4.5 Imperceptibility
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]
### 4.6 Label-agnostic Attack
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]
### 4.7 Poisoning against Defense
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]
### 4.8 Connection between Evasion Attack and Poisoning Attack
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]
### 4.9 Poisoning against Vision Transformer
- [2023] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]

- [2016] **Examining the impact of blur on recognition by convolutional networks** [[paper](https://arxiv.org/abs/1611.05760)]
### 4.10 Efficiency


  
