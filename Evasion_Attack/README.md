# EVASION ATTACK
  - [1 Non-adversarial Perturbation-based Evasion Attack](#1-Non-adversarial-Perturbation-based-Evasion-Attack)
    - [1.1 Current Studies and Underlying Reasons of Vulnerability for CNNs](#11-Current-Studies-and-Underlying-Reasons-of-Vulnerability-for-CNNs)
    - [1.2 Attack against Vision Transformers](#12-Attack-against-Vision-Transformers)
    - [1.3 Comparison between CNNs and ViTs](#13-Comparison-between-CNNs-and-ViTs)
    - [1.4 Effectiveness](#14-Effectiveness)
    - [1.5 Imperceptibility](#15-Imperceptibility)
  - [2 Non-adversarial Perturbation-based Poisoning Attack](#2-Non-adversarial-Perturbation-based-Poisoning-Attack)
    - [2.1 Current Studies and Underlying Reasons of Vulnerability](#21-Current-Studies-and-Underlying-Reasons-of-Vulnerability)
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
    - [4.3 Untargeted (Availability, Delusive, Indiscriminate) Attack](#43-Untargeted-Attack)
    - [4.4 Transferability](#34-Transferability)
      - [4.4.1 Downstream-agnostic Attack](#441-Downstream-agnostic-Attack)
        - [4.4.1.1 Targeted poisoning](#4411-Targeted-poisoning)
        - [4.4.1.2 Backdoor attacks](#4412-Backdoor-attacks)
        - [4.4.1.3 Untargeted attacks](#4413-Untargeted-attacks)
      - [4.5 Imperceptibility](#35-Imperceptibility)
      - [4.6 Label-agnostic Attack](#46-Label-agnostic-Attack)
      - [4.7 Poisoning against Defense](#47-Poisoning-against-Defense)
      - [4.8 Connection between Evasion Attack and Poisoning Attack](#48-Connection-between-Evasion-Attack-and-Poisoning-Attack)
      - [4.9 Poisoning against Vision Transformer](#49-Poisoning-against-Vision-Transformer)
      - [4.10 Efficiency](#410-Efficiency)
  <!-- - [Citation](#citation) -->

# PAPER LIST

## 1 Non-adversarial Perturbation-based Evasion Attack

### 1.1 Current Studies and Underlying Reasons of Vulnerability for CNNs

- [2023/10] **Language Models Hallucinate, but May Excel at Fact Verification** *Jian Guan et al. arXiv.* [[paper](https://arxiv.org/abs/2310.14564)]

### 1.2 Attack against Vision Transformers

- [2023/09] **FactLLaMA: Optimizing Instruction-Following Language Models with External Knowledge for Automated Fact-Checking** *Tsun-Hin Cheung and Kin-Man Lam. APSIPA ASC 2023.* [[paper](https://arxiv.org/abs/2309.00240)]
- [2023/07] **FacTool: Factuality Detection in Generative AI -- A Tool Augmented Framework for Multi-Task and Multi-Domain Scenarios** *I-Chun Chern et al. arXiv.* [[paper](https://arxiv.org/abs/2307.13528)]

### 1.3 Comparison between CNNs and ViTs
- [2023/09] **Bad Actor, Good Advisor: Exploring the Role of Large Language Models in Fake News Detection** *Beizhe Hu et al. arXiv.* [[paper](https://arxiv.org/abs/2309.12247)]
- [2023/09] **Detecting Misinformation with LLM-Predicted Credibility Signals and Weak Supervision** *João A. Leite et al. arxiv.* [[paper](https://arxiv.org/abs/2309.07601)]
- [2023/09] **Improving Multiclass Classification of Fake News Using BERT-Based Models and ChatGPT-Augmented Data** *Elena Shushkevich et al. MDPI Inventions.* [[paper](https://www.mdpi.com/2411-5134/8/5/112)]
- [2023/09] **Can Large Language Models Enhance Fake News Detection?: Improving Fake News Detection With Data Augmentation** *Emil Ahlbäck, Max Dougly.* [[paper](https://kth.diva-portal.org/smash/get/diva2:1778369/FULLTEXT01.pdf)]
- [2022/03] **Faking Fake News for Real Fake News Detection: Propaganda-loaded Training Data Generation** *Kung-Hsiang Huang et al. ACL 2023.* [[paper](https://arxiv.org/abs/2203.05386)]
  
### 1.4 Effectiveness

### 1.5 Imperceptibility


## 2 Non-adversarial Perturbation-based Poisoning Attack

- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]
- [2023/04] **Reinforcement Learning-Based Counter-Misinformation Response Generation: A Case Study of COVID-19 Vaccine Misinformation** *Bing He et al. WWW 2023.* [[paper](https://dl.acm.org/doi/10.1145/3543507.3583388)]
- [2023/04] **Working With AI to Persuade: Examining a Large Lan- guage Model’s Ability to Generate Pro-Vaccination Messages** *Elise Karinshak et al. CSCW 2023.* [[paper](https://dl.acm.org/doi/10.1145/3579592)]

#### 2.1 Current Studies and Underlying Reasons of Vulnerability
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

### 3 Adversarial Perturbation-based Evasion Attack

#### 3.1 Basic Generation Methods
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

#### 3.2 Underlying Reasons of Vulnerability
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

#### 3.3 Black-box Attack
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

##### 3.3.1 Query-based Attack
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

##### 3.3.2 Query Efficiency
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

##### 3.3.3 Transfer-based Attack and Adversarial Transferability (Underlying Reasons)
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

##### 3.3.4 Adversarial Transferability Enhancement
###### 3.3.4.1 Data augmentation
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]
###### 3.3.4.2 Ensemble-based techniques
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]
###### 3.3.4.3 Momentum-based methods
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]
###### 3.3.4.4 Architecture-oriented methods
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]
###### 3.3.4.5 Finding proper substitute models
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]
###### 3.3.4.6 Distribution-oriented methods
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]
###### 3.3.4.7 Other methods
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

##### 3.3.5 Cross-domain and modality Transferability
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]
- 
##### 3.3.6 Cross-task Transferability
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

#### 3.4 Perturbation against Vision Transformer and Cross-architecture Transferability
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

##### 3.4.1 Current Studies and Underlying Reasons for Vulnerability
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

##### 3.4.2 Comparison and Cross-architecture Transferability between CNNs and ViTs
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

#### 3.5 Non-box Attack
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

#### 3.6 Attack against Defense
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

#### 3.7 Problem of the Cross-entropy Loss
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

#### 3.8 Imperceptibility
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

#### 3.9 Diverse Perturbations
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

##### 3.9.1 Beyond l_p-norm Perturbations
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

##### 3.9.2 Beyond Single or Dual-type Perturbations
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

#### 3.10 Unconstrained Perturbations
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

## 4. Adversarial Perturbation-based Poisoning Attack

### 4.1 Targeted Poisoning Attack

- [2023/11] **Adapting Fake News Detection to the Era of Large Language Models.** *Jinyan Su et al. arXiv.* [[paper](https://arxiv.org/abs/2311.04917)]

### 4.2 Backdoor/Trojan Attack

#### 4.2.1 Alleviating Hallucination of LLMs

##### survey

- [2023/09] **Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models.** *Yue Zhang et al. arxiv.* [[paper](https://arxiv.org/abs/2309.01219)]
- [2023/09] **A Survey of Hallucination in Large Foundation Models.** *Vipula Rawte et al. arxiv.* [[paper](https://arxiv.org/abs/2309.05922)]

### 4.3 Untargeted (a.k.a. Availability, Delusive, Indiscriminate) Attack

### 4.4 Transferability
#### 4.4.1 Downstream-agnostic Attack

##### Targeted poisoning
- [2023/09] **Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models.** *Yue Zhang et al. arxiv.* [[paper](https://arxiv.org/abs/2309.01219)]

##### Backdoor attacks
- [2023/09] **Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models.** *Yue Zhang et al. arxiv.* [[paper](https://arxiv.org/abs/2309.01219)]

##### Untargeted attacks
- [2023/09] **Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models.** *Yue Zhang et al. arxiv.* [[paper](https://arxiv.org/abs/2309.01219)]

### 4.5 Imperceptibility

### 4.6 Label-agnostic Attack

### 4.7 Poisoning against Defense

### 4.8 Connection between Evasion Attack and Poisoning Attack

### 4.9 Poisoning against Vision Transformer

### 4.10 Efficiency


  
