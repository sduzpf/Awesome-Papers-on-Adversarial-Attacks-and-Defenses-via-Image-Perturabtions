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
        - [3.3.6.1 Downstream-agnostic Attack](#3361-Downstream-agnostic-Attack)
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
- [2024] **Foolmix: Strengthen the transferability of adversarial examples by dual-blending and direction update strategy** [[paper](https://ieeexplore.ieee.org/abstract/document/10508615/)]
- [2024] **Improving adversarial transferability through hybrid augmentation** [[paper](https://www.sciencedirect.com/science/article/pii/S0167404823005849)]
- [2024] **Enhancing the transferability of adversarial samples with random noise techniques** [[paper](https://www.sciencedirect.com/science/article/pii/S0167404823004510)]
- [2024] **Boosting adversarial transferability by block shuffle and rotation** [[paper](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Boosting_Adversarial_Transferability_by_Block_Shuffle_and_Rotation_CVPR_2024_paper.html)]
  
- [2023] **Improving the transferability of adversarial examples with arbitrary style transfer** [[paper](https://dl.acm.org/doi/abs/10.1145/3581783.3612070)]
- [2023] **Improving the transferability of adversarial samples by path-augmented method** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Improving_the_Transferability_of_Adversarial_Samples_by_Path-Augmented_Method_CVPR_2023_paper.html)]
- [2023] **Structure invariant transformation for better adversarial transferability** [[paper](http://openaccess.thecvf.com/content/ICCV2023/html/Wang_Structure_Invariant_Transformation_for_better_Adversarial_Transferability_ICCV_2023_paper.html)]
- [2023] **Enhancing the self-universality for transferable targeted attacks** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Wei_Enhancing_the_Self-Universality_for_Transferable_Targeted_Attacks_CVPR_2023_paper.html)]

- [2022] **Learning to learn transferable attack** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/19936)]
- [2022] **Adaptive image transformations for transfer-based adversarial attack** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-20065-6_1)]
- [2022] **Frequency domain model augmentation for adversarial attack** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-19772-7_32)]
  
- [2021] **Admix: Enhancing the transferability of adversarial attacks** [[paper](http://openaccess.thecvf.com/content/ICCV2021/html/Wang_Admix_Enhancing_the_Transferability_of_Adversarial_Attacks_ICCV_2021_paper.html)]


- [2019] **Improving transferability of adversarial examples with input diversity** [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Xie_Improving_Transferability_of_Adversarial_Examples_With_Input_Diversity_CVPR_2019_paper.html)]
- [2019] **Evading defenses to transferable adversarial examples by translation-invariant attacks** [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Dong_Evading_Defenses_to_Transferable_Adversarial_Examples_by_Translation-Invariant_Attacks_CVPR_2019_paper.html)]
- [2019] **Nesterov accelerated gradient and scale invariance for adversarial attacks** [[paper](https://arxiv.org/abs/1908.06281)]
  
###### 3.3.4.2 Ensemble-based techniques
- [2023] **An adaptive model ensemble adversarial attack for boosting adversarial transferability** [[paper](http://openaccess.thecvf.com/content/ICCV2023/html/Chen_An_Adaptive_Model_Ensemble_Adversarial_Attack_for_Boosting_Adversarial_Transferability_ICCV_2023_paper.html)]
- [2023] **Rethinking model ensemble in transfer-based adversarial attacks** [[paper](https://arxiv.org/abs/2303.09105)]
- [2023] **Minimizing maximum model discrepancy for transferable black-box targeted attacks** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Minimizing_Maximum_Model_Discrepancy_for_Transferable_Black-Box_Targeted_Attacks_CVPR_2023_paper.html)]
- [2022] **Learning to learn transferable attack** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/19936)]
- [2022] **Stochastic variance reduced ensemble adversarial attack for boosting the adversarial transferability** [[paper](http://openaccess.thecvf.com/content/CVPR2022/html/Xiong_Stochastic_Variance_Reduced_Ensemble_Adversarial_Attack_for_Boosting_the_Adversarial_CVPR_2022_paper.html)]
- [2019] **Evading defenses to transferable adversarial examples by translation-invariant attacks** [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Dong_Evading_Defenses_to_Transferable_Adversarial_Examples_by_Translation-Invariant_Attacks_CVPR_2019_paper.html)] 
- [2018] **Boosting adversarial attacks with momentum** [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.html)]
- [2016] **Delving into transferable adversarial examples and black-box attacks** [[paper](https://arxiv.org/abs/1611.02770)]

###### 3.3.4.3 Momentum-based methods
- [2018] **Boosting adversarial attacks with momentum** [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.html)]

- [2021] **Boosting adversarial transferability through enhanced momentum** [[paper](https://arxiv.org/abs/2103.10609)]

- [2024] **Improving adversarial transferability through frequency enhanced momentum** [[paper](https://www.sciencedirect.com/science/article/pii/S0020025524003220)]

- [2021] **Enhancing the transferability of adversarial attacks through variance tuning** [[paper](http://openaccess.thecvf.com/content/CVPR2021/html/Wang_Enhancing_the_Transferability_of_Adversarial_Attacks_Through_Variance_Tuning_CVPR_2021_paper.html)]

- [2024] **Nesterov accelerated gradient and scale invariance for adversarial attacks** [[paper](https://arxiv.org/abs/1908.06281)]

- [2019] **Transferable perturbations of deep feature distributions** [[paper](https://arxiv.org/abs/2004.12519)]
  
###### 3.3.4.4 Architecture-oriented methods
- [2019] **Skip connections matter: On the transferability of adversarial examples generated with ResNets** [[paper](https://arxiv.org/abs/2002.05990)]

- [2020] **Backpropagating linearly improves transferability of adversarial examples** [[paper](https://proceedings.neurips.cc/paper/2020/hash/00e26af6ac3b1c1c49d7c3d79c60d000-Abstract.html)]

- [2021] **Backpropagating smoothly improves transferability of adversarial examples** [[paper]([https://arxiv.org/abs/1611.05760](https://aisecure-workshop.github.io/amlcvpr2021/cr/31.pdf))]

- [2023] **Backpropagation path search on adversarial transferability** [[paper](http://openaccess.thecvf.com/content/ICCV2023/html/Xu_Backpropagation_Path_Search_On_Adversarial_Transferability_ICCV_2023_paper.html)]

- [2021] **Batch normalization increases adversarial vulnerability and decreases adversarial transferability: A non-robust feature perspective** [[paper](http://openaccess.thecvf.com/content/ICCV2021/html/Benz_Batch_Normalization_Increases_Adversarial_Vulnerability_and_Decreases_Adversarial_Transferability_A_ICCV_2021_paper.html)]

- [2020] **Boosting the transferability of adversarial samples via attention** [[paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Wu_Boosting_the_Transferability_of_Adversarial_Samples_via_Attention_CVPR_2020_paper.html)]

- [2021] **Feature importance-aware transferable adversarial attacks** [[paper]([https://arxiv.org/abs/1611.05760](http://openaccess.thecvf.com/content/ICCV2021/html/Wang_Feature_Importance-Aware_Transferable_Adversarial_Attacks_ICCV_2021_paper.html))]

###### 3.3.4.5 Finding proper substitute models
- [2024] **AGS: Affordable and generalizable substitute training for transferable adversarial attack** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28365)]

- [2023] **StyLess: Boosting the transferability of adversarial examples** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Liang_StyLess_Boosting_the_Transferability_of_Adversarial_Examples_CVPR_2023_paper.html)]
- [2023] **Diffusion models for imperceptible and transferable adversarial attack** [[paper](https://ieeexplore.ieee.org/abstract/document/10716799/)]
- [2023] **Why does little robustness help? a further step towards understanding adversarial transferability** [[paper](https://ieeexplore.ieee.org/abstract/document/10646840/)]
- [2023] **On the role of generalization in transferability of adversarial examples** [[paper](https://proceedings.mlr.press/v216/wang23g.html)]

- [2021] **A little robustness goes a long way: Leveraging robust features for targeted transfer attacks** [[paper](https://proceedings.neurips.cc/paper/2021/hash/50f3f8c42b998a48057e9d33f4144b8b-Abstract.html)]

- [2020] **Practical no-box adversarial attacks against dnns** [[paper]([https://arxiv.org/abs/1611.05760](https://proceedings.neurips.cc/paper/2020/hash/96e07156db854ca7b00b5df21716b0c6-Abstract.html))]
  
###### 3.3.4.6 Distribution-oriented methods

- [2023] **Towards verifying the geometric robustness of large-scale neural networks** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26773)]
  
- [2022] **Toward understanding and boosting adversarial transferability from a distribution perspective** [[paper](https://ieeexplore.ieee.org/abstract/document/9917370/)]
  
- [2021] **On generating transferable targeted perturbations** [[paper](http://openaccess.thecvf.com/content/ICCV2021/html/Naseer_On_Generating_Transferable_Targeted_Perturbations_ICCV_2021_paper.html)]
  
- [2020] **Perturbing across the feature hierarchy to improve standard and strict blackbox attack transferability** [[paper](https://proceedings.neurips.cc/paper/2020/hash/eefc7bfe8fd6e2c8c01aa6ca7b1aab1a-Abstract.html)]
- [2019] **Transferable perturbations of deep feature distributions** [[paper](https://arxiv.org/abs/2004.12519)]
###### 3.3.4.7 Other methods

- [2024] **Perturbation towards easy samples improves targeted adversarial transferability** [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html)]

- [2023] **Boosting adversarial transferability with learnable patch-wise masks** [[paper](https://ieeexplore.ieee.org/abstract/document/10251606/)]

- [2023] **Boosting adversarial transferability by achieving flat local maxima** [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/de1739eba209c682a90ec3669229ab2d-Abstract-Conference.html)]
- [2022] **Boosting the transferability of adversarial attacks with reverse adversarial perturbation** [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c0f9419caa85d7062c7e6d621a335726-Abstract-Conference.html)]

- [2020] **A unified approach to interpreting and boosting adversarial transferability** [[paper](https://arxiv.org/abs/2010.04055)]
- [2019] **Cross-domain transferability of adversarial perturbations** [[paper](https://proceedings.neurips.cc/paper_files/paper/2019/hash/99cd3843754d20ec3c5885d805db8a32-Abstract.html)]
  
##### 3.3.5 Cross-domain and modality Transferability
- [2024] **UCG: A universal cross-domain generator for transferable adversarial examples** [[paper](https://ieeexplore.ieee.org/abstract/document/10388391/)]
- [2024] **FACL-Attack: Frequency-aware contrastive learning for transferable adversarial attacks** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28470)]
- [2024] **AGS: Affordable and generalizable substitute training for transferable adversarial attack** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28365)]

- [2023] **Adaptive cross-modal transferable adversarial attacks from images to videos** [[paper](https://ieeexplore.ieee.org/abstract/document/10375740/)]
- [2023] **GCMA: Generative cross-modal transferable adversarial attacks from images to videos** [[paper](https://dl.acm.org/doi/abs/10.1145/3581783.3612110)]
- [2023] **Breaking temporal consistency: Generating video universal adversarial perturbations using image models** [[paper](http://openaccess.thecvf.com/content/ICCV2023/html/Kim_Breaking_Temporal_Consistency_Generating_Video_Universal_Adversarial_Perturbations_Using_Image_ICCV_2023_paper.html)]
- [2023] **Global-local characteristic excited cross-modal attacks from images to videos** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25362)]
- [2023] **CDTA: A cross-domain transfer-based attack with contrastive learning** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25239)]
  
- [2022] **Cross-modal transferable adversarial attacks from images to videos** [[paper](http://openaccess.thecvf.com/content/CVPR2022/html/Wei_Cross-Modal_Transferable_Adversarial_Attacks_From_Images_to_Videos_CVPR_2022_paper.html)]
- [2021] **Learning transferable adversarial perturbations** [[paper](https://proceedings.neurips.cc/paper/2021/hash/7486cef2522ee03547cfb970a404a874-Abstract.html)]


##### 3.3.6 Cross-task Transferability

- [2024] **Enhancing cross-task transferability of adversarial examples via spatial and channel attention** [[paper](https://ieeexplore.ieee.org/abstract/document/10399858/)]

- [2023] **An image is worth 1000 lies: Transferability of adversarial images across prompts on vision-language models** [[paper](https://arxiv.org/abs/1611.05760)]

- [2020] **Enhancing cross-task black-box transferability of adversarial examples with dispersion reduction** [[paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Lu_Enhancing_Cross-Task_Black-Box_Transferability_of_Adversarial_Examples_With_Dispersion_Reduction_CVPR_2020_paper.html)]

##### 3.3.6.1 Downstream-agnostic Attack

- [2023] **Downstream-agnostic adversarial examples** [[paper](http://openaccess.thecvf.com/content/ICCV2023/html/Zhou_Downstream-agnostic_Adversarial_Examples_ICCV_2023_paper.html)]
  
- [2022] **Pre-trained adversarial perturbations** [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/084727e8abf90a8365b940036329cb6f-Abstract-Conference.html)]


##### 3.3.6.2 Multi-modal Downstream-agnostic Attack
- [2024] **VLATTACK: Multimodal adversarial attacks on vision-language tasks via pre-trained models** [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a5e3cf29c269b041ccd644b6beaf5c42-Abstract-Conference.html)]
- [2024] **Universal adversarial perturbations for vision-language pre-trained models** [[paper](https://dl.acm.org/doi/abs/10.1145/3626772.3657781)]

- [2023] **Set-level guidance attack: Boosting adversarial transferability of vision-language pre-training models** [[paper](http://openaccess.thecvf.com/content/ICCV2023/html/Lu_Set-level_Guidance_Attack_Boosting_Adversarial_Transferability_of_Vision-Language_Pre-training_Models_ICCV_2023_paper.html)]

- [2023] **SA-Attack: Improving adversarial transferability of vision-language pre-training models via self-augmentation** [[paper](https://arxiv.org/abs/2312.04913)]

- [2023] **Advclip: Downstream-agnostic adversarial examples in multimodal contrastive learning** [[paper](https://dl.acm.org/doi/abs/10.1145/3581783.3612454)]

- [2023] **Downstream task-agnostic transferable attacks on language-image pre-training models** [[paper](https://ieeexplore.ieee.org/abstract/document/10219910/)]


- [2022] **Towards adversarial attack on vision-language pre-training models** [[paper](https://dl.acm.org/doi/abs/10.1145/3503161.3547801)]
  
#### 3.4 Perturbation against Vision Transformer and Cross-architecture Transferability


##### 3.4.1 Current Studies and Underlying Reasons for Vulnerability
- [2024] **Improving the adversarial transferability of vision transformers with virtual dense connection** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28541)]

- [2023] **Transferable adversarial attacks on vision transformers with token gradient regularization** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Transferable_Adversarial_Attacks_on_Vision_Transformers_With_Token_Gradient_Regularization_CVPR_2023_paper.html)]
- [2023] **Boosting adversarial transferability with learnable patch-wise masks** [[paper](https://ieeexplore.ieee.org/abstract/document/10251606/)]
- [2023] **Improving robustness of vision transformers by reducing sensitivity to patch corruptions** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Guo_Improving_Robustness_of_Vision_Transformers_by_Reducing_Sensitivity_To_Patch_CVPR_2023_paper.html)]
- [2023] **Query-efficient decision-based black-box patch attack** [[paper](https://ieeexplore.ieee.org/abstract/document/10227335/)]

- [2022] **Generating transferable adversarial examples against vision transformers** [[paper](https://dl.acm.org/doi/abs/10.1145/3503161.3547989)]
- [2022] **Towards transferable adversarial attacks on vision transformers** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20169)]
- [2022] **Towards robust vision transformer** [[paper](http://openaccess.thecvf.com/content/CVPR2022/html/Mao_Towards_Robust_Vision_Transformer_CVPR_2022_paper.html)]
  
- [2021] **Understanding robustness of transformers for image classification** [[paper](http://openaccess.thecvf.com/content/ICCV2021/html/Bhojanapalli_Understanding_Robustness_of_Transformers_for_Image_Classification_ICCV_2021_paper.html)]
- [2021] **Beware the black-box: On the robustness of recent defenses to adversarial examples** [[paper](https://www.mdpi.com/1099-4300/23/10/1359)]
- [2021] **On improving adversarial transferability of vision transformers** [[paper](https://arxiv.org/abs/2106.04169)]
  
- [2019] **Exploring the landscape of spatial robustness** [[paper](https://arxiv.org/abs/1611.05760)]

##### 3.4.2 Comparison and Cross-architecture Transferability between CNNs and ViTs

- [2024] **Ensemble diversity facilitates adversarial transferability** [[paper](https://openaccess.thecvf.com/content/CVPR2024/html/Tang_Ensemble_Diversity_Facilitates_Adversarial_Transferability_CVPR_2024_paper.html)]
- [2024] **UCG: A universal cross-domain generator for transferable adversarial examples** [[paper](https://ieeexplore.ieee.org/abstract/document/10388391/)]
- [2024] **Attacking transformers with feature diversity adversarial perturbation** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/27947)]
  
- [2023] **Exploring the differences in adversarial robustness between ViT-and CNN-based models using novel metrics** [[paper](https://www.sciencedirect.com/science/article/pii/S1077314223001807)]
- [2023] **Understanding and improving adversarial transferability of vision transformers and convolutional neural networks** [[paper](https://www.sciencedirect.com/science/article/pii/S0020025523010599)]
- [2023] **Transferable adversarial attack for both vision transformers and convolutional networks via momentum integrated gradients** [[paper](http://openaccess.thecvf.com/content/ICCV2023/html/Ma_Transferable_Adversarial_Attack_for_Both_Vision_Transformers_and_Convolutional_Networks_ICCV_2023_paper.html)]
  
- [2021] **On the robustness of vision transformers to adversarial examples** [[paper](http://openaccess.thecvf.com/content/ICCV2021/html/Mahmood_On_the_Robustness_of_Vision_Transformers_to_Adversarial_Examples_ICCV_2021_paper.html)]
- [2021] **On improving adversarial transferability of vision transformers** [[paper](https://arxiv.org/abs/2106.04169)]
- [2021] **Are transformers more robust than cnns?** [[paper](https://proceedings.neurips.cc/paper/2021/hash/e19347e1c3ca0c0b97de5fb3b690855a-Abstract.html)]
- [2021] **Intriguing properties of vision transformers** [[paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/c404a5adbf90e09631678b13b05d9d7a-Abstract.html)]

- [2020] **High-frequency component helps explain the generalization of convolutional neural networks** [[paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Wang_High-Frequency_Component_Helps_Explain_the_Generalization_of_Convolutional_Neural_Networks_CVPR_2020_paper.html)]
  
#### 3.5 Non-box Attack
- [2021] **Data-free universal adversarial perturbation and black-box attack** [[paper](http://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Data-Free_Universal_Adversarial_Perturbation_and_Black-Box_Attack_ICCV_2021_paper.html)]
- [2020] **Practical no-box adversarial attacks against dnns** [[paper](https://proceedings.neurips.cc/paper/2020/hash/96e07156db854ca7b00b5df21716b0c6-Abstract.html)]


#### 3.6 Attack against Defense
- [2024] **DiffAttack: Evasion attacks against diffusion-based adversarial purification** [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ea0b28cbbd0cbc45ec4ac38e92da9cb2-Abstract-Conference.html)]
- [2021] **Beware the black-box: On the robustness of recent defenses to adversarial examples** [[paper](https://www.mdpi.com/1099-4300/23/10/1359)]

- [2020] **On adaptive attacks to adversarial example defenses** [[paper](https://proceedings.neurips.cc/paper/2020/hash/11f38f8ecd71867b42433548d1078e38-Abstract.html)]
- [2020] **Colorfool: Semantic adversarial colorization** [[paper](https://arxiv.org/abs/1611.05760)]
- [2019] **Breaking certified defenses: Semantic adversarial examples with spoofed robustness certificates** [[paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Shamsabadi_ColorFool_Semantic_Adversarial_Colorization_CVPR_2020_paper.html)]
- [2018] **Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples** [[paper](http://proceedings.mlr.press/v80/athalye18a.html)]

#### 3.7 Problem of the Cross-entropy Loss
- [2024] **Towards understanding and improving adversarial robustness of vision transformers** [[paper](https://openaccess.thecvf.com/content/CVPR2024/html/Jain_Towards_Understanding_and_Improving_Adversarial_Robustness_of_Vision_Transformers_CVPR_2024_paper.html)]

- [2023] **Efficient loss function by minimizing the detrimental effect of floating-point errors on gradient-based attacks** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Yu_Efficient_Loss_Function_by_Minimizing_the_Detrimental_Effect_of_Floating-Point_CVPR_2023_paper.html)]

- [2023] **Logit margin matters: Improving transferable targeted adversarial attack by logit calibration** [[paper](https://ieeexplore.ieee.org/abstract/document/10147340/)]
- [2021] **On success and simplicity: A second look at transferable targeted attacks** [[paper](https://proceedings.neurips.cc/paper/2021/hash/30d454f09b771b9f65e3eaf6e00fa7bd-Abstract.html)]

- [2020] **Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks** [[paper](https://proceedings.mlr.press/v119/croce20b.html)]

- [2020] **Towards transferable targeted attack** [[paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Li_Towards_Transferable_Targeted_Attack_CVPR_2020_paper.html)]
- [2020] **Universal adversarial training** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6017)]

- [2019] **Defending against universal perturbations with shared adversarial training** [[paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Mummadi_Defending_Against_Universal_Perturbations_With_Shared_Adversarial_Training_ICCV_2019_paper.html)]

- [2019] **Decoupling direction and norm for efficient gradient-based l2 adversarial attacks and defenses** [[paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Rony_Decoupling_Direction_and_Norm_for_Efficient_Gradient-Based_L2_Adversarial_Attacks_CVPR_2019_paper.html)]

- [2018] **Boosting adversarial attacks with momentum** [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.html)]

- [2017] **Towards evaluating the robustness of neural networks** [[paper](https://ieeexplore.ieee.org/abstract/document/7958570/)]

  
#### 3.8 Imperceptibility
- [2023] **Towards verifying the geometric robustness of large-scale neural networks** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26773)]
- [2023] **AdvST: Generating unrestricted adversarial images via style transfer** [[paper](https://ieeexplore.ieee.org/abstract/document/10292904/)]
- [2023] **Diffusion models for imperceptible and transferable adversarial attack** [[paper](https://ieeexplore.ieee.org/abstract/document/10716799/)]
- [2023] **Advdiffuser: Natural adversarial example synthesis with diffusion models** [[paper](http://openaccess.thecvf.com/content/ICCV2023/html/Chen_AdvDiffuser_Natural_Adversarial_Example_Synthesis_with_Diffusion_Models_ICCV_2023_paper.html)]
- [2023] **Content-based unrestricted adversarial attack** [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a24cd16bc361afa78e57d31d34f3d936-Abstract-Conference.html)]
- [2023] **Alias-free convnets: Fractional shift invariance via polynomial activations** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Michaeli_Alias-Free_Convnets_Fractional_Shift_Invariance_via_Polynomial_Activations_CVPR_2023_paper.html)]

- [2022] **Natural color fool: Towards boosting black-box unrestricted attacks** [[paper]([https://arxiv.org/abs/1611.05760](https://proceedings.neurips.cc/paper_files/paper/2022/hash/31d0d59fe946684bb228e9c8e887e176-Abstract-Conference.html))]

- [2020] **A self-supervised approach for adversarial robustness** [[paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Naseer_A_Self-supervised_Approach_for_Adversarial_Robustness_CVPR_2020_paper.html)]
- [2020] **Colorfool: Semantic adversarial colorization** [[paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Shamsabadi_ColorFool_Semantic_Adversarial_Colorization_CVPR_2020_paper.html)]

- [2019] **Exploring the landscape of spatial robustness** [[paper](http://proceedings.mlr.press/v97/engstrom19a.html)]
- [2019] **Unrestricted adversarial examples via semantic manipulation** [[paper](https://arxiv.org/abs/1904.06347)]

- [2018] **Geometric robustness of deep networks: Analysis and improvement** [[paper]([https://arxiv.org/abs/1611.05760](http://openaccess.thecvf.com/content_cvpr_2018/html/Kanbak_Geometric_Robustness_of_CVPR_2018_paper.html))]
- [2018] **Beyond pixel norm-balls: Parametric adversaries using an analytically differentiable renderer** [[paper](https://arxiv.org/abs/1808.02651)]
- [2018] **Semantic adversarial examples** [[paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w32/html/Hosseini_Semantic_Adversarial_Examples_CVPR_2018_paper.html)]

#### 3.9 Diverse Perturbations
##### 3.9.1 Beyond l_p-norm Perturbations
- [2024] **Wasserstein distributional robustness of neural networks** [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/53be3798fcc46e68ca0819c29a004652-Abstract-Conference.html)]
- [2022] **Sinkhorn adversarial attack and defense** [[paper](https://ieeexplore.ieee.org/abstract/document/9792616/)]
  
- [2021] **Augmented lagrangian adversarial attacks** [[paper](http://openaccess.thecvf.com/content/ICCV2021/html/Rony_Augmented_Lagrangian_Adversarial_Attacks_ICCV_2021_paper.html)]
- [2021] **Perceptual quality-preserving black-box attack against deep learning image classifiers** [[paper](https://www.sciencedirect.com/science/article/pii/S0167865521001288)]
- [2021] **Perceptual adversarial robustness: Defense against unseen threat models** [[paper](https://arxiv.org/abs/2006.12655)]

- [2020] **Towards large yet imperceptible adversarial image perturbations with perceptual color distance** [[paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Zhao_Towards_Large_Yet_Imperceptible_Adversarial_Image_Perturbations_With_Perceptual_Color_CVPR_2020_paper.html)]
- [2020] **Stronger and faster wasserstein adversarial attacks** [[paper](https://proceedings.mlr.press/v119/wu20d.html)]

- [2019] **Wasserstein adversarial examples via projected sinkhorn iterations** [[paper](http://proceedings.mlr.press/v97/wong19a)]
- [2018] **Geometric robustness of deep networks: Analysis and improvement** [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Kanbak_Geometric_Robustness_of_CVPR_2018_paper.html)]
- [2018] **Exploring the landscape of spatial robustness** [[paper](http://proceedings.mlr.press/v97/engstrom19a.html)]


##### 3.9.2 Beyond Single or Dual-type Perturbations
- [2021] **Fast minimum-norm adversarial attacks through adaptive norm constraints** [[paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/a709909b1ea5c2bee24248203b1728a5-Abstract.html)]
  
- [2020] **Minimally distorted adversarial examples with a fast adaptive boundary attack** [[paper](https://proceedings.mlr.press/v119/croce20a.html)]
- [2020] **Accurate, reliable and fast robustness evaluation** [[paper](https://proceedings.neurips.cc/paper/2019/hash/885fe656777008c335ac96072a45be15-Abstract.html)]

#### 3.10 Unconstrained Perturbations
- [2023] **Diffusion models for imperceptible and transferable adversarial attack** [[paper](https://ieeexplore.ieee.org/abstract/document/10716799/)]
- [2023] **Advdiffuser: Natural adversarial example synthesis with diffusion models** [[paper](http://openaccess.thecvf.com/content/ICCV2023/html/Chen_AdvDiffuser_Natural_Adversarial_Example_Synthesis_with_Diffusion_Models_ICCV_2023_paper.html)]
- [2023] **Content-based unrestricted adversarial attack** [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a24cd16bc361afa78e57d31d34f3d936-Abstract-Conference.html)]
- [2023] **AdvST: Generating unrestricted adversarial images via style transfer** [[paper](https://ieeexplore.ieee.org/abstract/document/10292904/)]
- [2023] **Alias-free convnets: Fractional shift invariance via polynomial activations** [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Michaeli_Alias-Free_Convnets_Fractional_Shift_Invariance_via_Polynomial_Activations_CVPR_2023_paper.html)]

- [2023] **Towards verifying the geometric robustness of large-scale neural networks** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26773)]
- [2022] **Natural color fool: Towards boosting black-box unrestricted attacks** [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/31d0d59fe946684bb228e9c8e887e176-Abstract-Conference.html)]

- [2020] **Colorfool: Semantic adversarial colorization** [[paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Shamsabadi_ColorFool_Semantic_Adversarial_Colorization_CVPR_2020_paper.html)]
- [2020] **Adversarial t-shirt! evading person detectors in a physical world** [[paper](https://link.springer.com/chapter/10.1007/978-3-030-58558-7_39)]
- [2019] **Exploring the landscape of spatial robustness** [[paper](http://proceedings.mlr.press/v97/engstrom19a.html)]
- [2019] **Unrestricted adversarial examples via semantic manipulation** [[paper]([https://arxiv.org/abs/1611.05760](https://arxiv.org/abs/1904.06347))]
- [2018] **Robust physical-world attacks on deep learning visual classification** [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Eykholt_Robust_Physical-World_Attacks_CVPR_2018_paper)]
- [2018] **CAMOU: Learning physical vehicle camouflages to adversarially attack detectors in the wild** [[paper](https://openreview.net/forum?id=SJgEl3A5tm)]
- [2018] **Geometric robustness of deep networks: Analysis and improvement** [[paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Kanbak_Geometric_Robustness_of_CVPR_2018_paper.html)]
- [2018] **Beyond pixel norm-balls: Parametric adversaries using an analytically differentiable renderer** [[paper](https://arxiv.org/abs/1808.02651)]
- [2018] **Semantic adversarial examples** [[paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w32/html/Hosseini_Semantic_Adversarial_Examples_CVPR_2018_paper.html)]
- [2018] **Lavan: Localized and visible adversarial noise** [[paper](https://proceedings.mlr.press/v80/karmon18a.html)]
- [2017] **Adversarial patch** [[paper](https://arxiv.org/abs/1712.09665)]
- [2016] **Accessorize to a crime: Real and stealthy attacks on state-of-the-art face recognition** [[paper](https://dl.acm.org/doi/abs/10.1145/2976749.2978392)]

