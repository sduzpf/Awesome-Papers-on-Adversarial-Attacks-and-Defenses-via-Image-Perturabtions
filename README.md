# Paper List for  Image Perturbations for Model Robustness: Attacks and Defenses# llm-misinformation-survey

<div align=center><img src="./image/logo.png" width="90%"/></div>

>The repository for the survey **[A Survey on Image Perturbations for Model Robustness: Attacks and Defenses](https://www.researchgate.net/profile/Pengfei-Zhang-74/publication/385213781_A_Survey_on_Image_Perturbations_for_Model_Robustness_Attacks_and_Defenses/links/671af1e4393e8533f715a9f2/A-Survey-on-Image-Perturbations-for-Model-Robustness-Attacks-and-Defenses.pdf)**  <br> **TLDR** : A survey of image perturbations and their influence on model robustness via adversarial attacks and defenses.  <br>

>Any suggestion, comment or related discussion is welcome. 

If you find our survey or paper list useful, we would greatly appreciate it if you could consider citing our paper:
```
@article{zhangsurvey,
  title={A Survey on Image Perturbations for Model Robustness: Attacks and Defenses},
  author={Zhang, Peng-Fei and Huang, Zi}
  year = {2024},
}
```

## Table of Content (ToC)

- [image-perturbations-survey](#image-perturbations-survey)
  - [2. PERTURBATION-BASED ATTACK](./Adversarial_Attack/README.md)
    - [2.1 Non-adversarial Perturbation-based Evasion Attack](#21-Non-adversarial_Perturbation-based_Evasion_Attack)
      - [2.1.1 Current Studies and Underlying Reasons of Vulnerability for CNNs](#211-Current_Studies_for_CNNs)
      - [2.1.2 Attack against Vision Transformers](#212-ttack against Vision Transformers)
      - [2.1.3 Comparison between CNNs and ViTs](#213-Comparison between CNNs and ViTs)
      - [2.1.4 Effectiveness](#213-Effectiveness)
      - [2.1.5 Imperceptibility](#215-Imperceptibility)
    - [2.2 Non-adversarial Perturbation-based Poisoning Attack](#22-on-adversarial Perturbation-based Poisoning Attack)
      - [2.2.1 Current Studies and Underlying Reasons of Vulnerability](#221-Current Studies)
    - [2.3 Adversarial Perturbation-based Evasion Attack](#23-Adversarial Perturbation-based Evasion Attack)
      - [2.3.1 Basic Generation Methods](#231-Basic Generation Methods)
      - [2.3.2 Underlying Reasons of Vulnerability](#232-Underlying Reasons of Vulnerability)
      - [2.3.3 Black-box Attack](#233-Black-box Attack)
          - [2.3.3.1 Query-based Attack](#2331-Query-based Attack)
          - [2.3.3.2 Query Efficiency](#2332-Query Efficiency)
          - [2.3.3.3 Transfer-based Attack and Adversarial Transferabily](#2333-Transfer-based Attack and Adversarial Transferability)
            - [2.3.3.3.1 Underlying Reasons for Adversarial Transferability](#23331-Underlying Reasons for Adversarial Transferability)
          - [2.3.3.4 Adversarial Transferability Enhancement](#2334-Adversarial Transferability Enhancement)
            - [2.3.3.4.1 Data augmentation](#23341-Data augmentation)
            - [2.3.3.4.2 Ensemble-based techniques](#23342-Ensemble-based techniques)
            - [2.3.3.4.3 Momentum-based methods](#23343-Momentum-based methods)
            - [2.3.3.4.4 Architecture-oriented methods](#23344-Architecture-oriented methods)
            - [2.3.3.4.5 Finding proper substitute models](#23345-Finding proper substitute models)
            - [2.3.3.4.6 Distribution-oriented methods](#23346-Distribution-oriented methods)
            - [2.3.3.4.7 Other methods](#23347-Other methods)
          - [2.3.3.5 Cross-domain/modality Transferability](#2335-Cross-domain/modality Transferability)
          - [2.3.3.6 Cross-task Transferability](#2336-Cross-task Transferability)
        - [2.3.4 Perturbation against Vision Transformer and Cross-architecture Transferability](#234-Perturbation against Vision Transformer and Cross-
architecture Transferability)
          - [2.3.4.1 Current studies and Underlying Reasons for Vulnerability](#2341-Current studies and Underlying Reasons for Vulnerability)
          - [2.3.4.2 Comparison and Cross-architecture Transferability between CNNs and ViTs](#2342-Comparison and Cross-architecture Transferability between CNNs and ViTs)
        - [2.3.5 Non-box Attack](#235-Non-box Attack)
        - [2.3.6 Attack against Defense](#236-Attack against Defense)
        - [2.3.7 Problem of the Cross-entropy Loss](#237-Problem of the Cross-entropy Loss)
        - [2.3.8 Imperceptibility](#238-Imperceptibility)
        - [2.3.9 Diverse Perturbations](#239-Diverse Perturbations)
          - [2.3.9.1 Beyond l_p-norm Perturbations](#2391-Beyond l_p-norm Perturbations)
          - [2.3.9.2 Beyond Single or Dual-type Perturbations](#2392-Beyond Single or Dual-type Perturbations)
        - [2.3.10 Unconstrained Perturbations](#2310-Unconstrained Perturbations)
  - [3. Adversarial Perturbation-based Poisoning Attack](#3-Adversarial Perturbation-based Poisoning Attack)
    - [3.1 Targeted Poisoning Attack](#31-Targeted Poisoning Attack)
    - [3.2 Backdoor/Trojan Attack](#32-Backdoor/Trojan Attack)
      - [3.2.1 Alleviating Hallucination of LLMs](#321-alleviating-hallucination-of-llms)
      - [3.2.2 Improving Safety of LLMs](#322-improving-safety-of-llms)
      - [3.2.3 Detecting LLM-Generated Misinformation](#323-detecting-llm-generated-misinformation)
    - [3.3 Untargeted (a.k.a. Availability, Delusive, Indiscriminate) Attack](#33-Untargeted (a.k.a. Availability, Delusive, Indiscriminate) Attack)
    - [3.4 Transferability](#34-Transferability)
      - [3.4.1 Downstream-agnostic Attack](#341-Downstream-agnostic Attack)
        - [3.4.1.1 Targeted poisoning](#3411-Targeted poisoning)
        - [3.4.1.2 Backdoor attacks](#3412-Backdoor attacks)
        - [3.4.1.3 Untargeted attacks](#3413-Untargeted attacks)
      - [3.5 Imperceptibility](#35-Imperceptibility)
      - [3.6 Label-agnostic Attack](#36-Label-agnostic Attack)
      - [3.7 Poisoning against Defense](#37-Poisoning against Defense)
      - [3.8 Connection between Evasion Attack and Poisoning Attack](#38-Connection between Evasion Attack and Poisoning Attack)
      - [3.9 Poisoning against Vision Transformer](#39-Poisoning against Vision Transformer)
      - [3.10 Efficiency](#310-Efficiency)
  - [4. Perturbation-based Augmentation](#4-Perturbation-based Augmentation)
      - [4.1 Defend against Non-adversarial Perturbation-based Evasion Attack](#41-Defend against Non-adversarial Perturbation-based Evasion Attack)
        - [4.1.1 Current Methods](#411-Current Methods)
        - [4.1.2 Perturbation-based Augmentation for ViTs](#412-Perturbation-based Augmentation for ViTs)
        - [4.1.3 Comparison between CNNs and ViTs](#413-Comparison between CNNs and ViTs)
      - [4.2 Defense against Adversarial Perturbation-based Evasion Attack](#42-Defense against Adversarial Perturbation-based Evasion Attack)
      - [4.3 Defense against Non-adversarial Perturbation-based Poisoning Attack](#43-Defense against Non-adversarial Perturbation-based Poisoning Attack)
      - [4.4 Defense against Adversarial Perturbation-based Poisoning Attack](#44-Defense against Adversarial Perturbation-based Poisoning Attack)
  - [5. Perturbation-based Input Pre-processing](#5-Perturbation-based Input Pre-processing)
      - [5.1 Defense against Adversarial Perturbation-based Evasion Attack](#51-Defense against Adversarial Perturbation-based Evasion Attack)
      - [5.2 Defense against Adversarial Perturbation-based Poisoning Attack](#52-Defense against Adversarial Perturbation-based Poisoning Attack)
  - [6. Randomized Smoothing for Certified Adversarial Robustness](#5-Randomized Smoothing for Certified Adversarial Robustness)
      - [6.1 Basic Learning Methods](#61-Basic Learning Methods)
      - [6.2 Diverse Perturbations](#62-Diverse Perturbations)
        - [6.2.1 Beyond l_2-norm Robustness Certification](#621-Beyond l_2-norm Robustness Certification)
        - [6.2.2 Beyond Gaussian Noise](#622-Beyond Gaussian Noise)
      - [6.3 Certified Robustness and its Trade-off with Standard Accuracy](#63-Certified Robustness and its Trade-off with Standard Accuracy)
      - [6.4 Efficiency](#64-Efficiency)
      - [6.5 Curse of Dimensionality](#65-Curse of Dimensionality)
      - [6.6 Comparison and Connection between Adversarial Training and Randomized Smoothing](#66-Comparison and Connection between Adversarial Training and Randomized Smoothing)
      - [6.7 Defend against Non-adversarial Perturbation-based Evasion Attack](#67-Defend against Non-adversarial Perturbation-based Evasion Attack)
   - [7. Adversarial Training](#7-Adversarial Training)
      - [7.1 Definition](#71-Definition)
      - [7.2 Overfitting (a.k.a. Generalization)](#72-Overfitting (a.k.a. Generalization))
        - [7.2.1 Catastrophic Overfitting](#721-Catastrophic Overfitting)
          - [7.2.1.1 Underlying Reasons](#7211-Underlying Reasons)
          - [7.2.1.2 Solutions](#7212-Solutions)  
      - [7.2.2 Robust Overfitting](#722-Robust Overfitting)
          - [7.2.2.1 Underlying Reasons](#7221-Underlying Reasons)
          - [7.2.2.2 Solutions](#7222-Solutions)  
      - [7.3 Adversarial Robustness Enhancement](#73-Adversarial Robustness Enhancement)
      - [7.4 Robust Fairness](#74-Robust Fairness)
      - [7.5 Trade-off between Adversarial Robustness and Standard Accuracy](#75-Trade-off between Adversarial Robustness and Standard Accuracy)
      - [7.6 Comparison and Connection between Adversarial Training and Randomized Smoothing](#76-Comparison and Connection between Adversarial Training and Randomized Smoothing)
      - [7.7 Defense against Patch Attack](#77-Defense against Patch Attack)
      - [7.8 Multi-Attack Robustness](#78-Multi-Attack Robustness)
      - [7.9 Cross-network/task Adversarial Training](#79-Cross-network/task Adversarial Training)
      - [7.10 Robust Pre-training and Fine-tuning](#710-Robust Pre-training and Fine-tuning)
      - [7.11 Adaptive Perturbations](#711-Adaptive Perturbations)
      - [7.12 Efficiency](#712-Efficiency)
      - [7.13 Adversarial Training for ViTs and Comparison with CNNs](#713-Adversarial Training for ViTs and Comparison with CNNs)
      - [7.14 Adversarial Training against Poisoning Attack](#714-Adversarial Training against Poisoning Attack)
        - [7.14.1 Adversarial Training against Backdoor Attack](#7141-Adversarial Training against Backdoor Attack)
        - [7.14.2 Adversarial Training against Availability Attack](#7142-Adversarial Training against Availability Attack)
        - [6.2.2 Beyond Gaussian Noise](#622-Beyond Gaussian Noise)
    - [8. Relation between different attacks]
      - [8.1 Relations between Different Adversarial Robustness to Evasion Attack](#81-Relations between Different Adversarial Robustness to Evasion Attack)
      - [8.2 Relations between Adversarial Robustness to Evasion Attack and Backdoor Attack](#82-Relations between Adversarial Robustness to Evasion Attack and Backdoor Attack)
      - [8.3 Relations between Adversarial and Non-adversarial Robustness](#83-Relations between Adversarial and Non-adversarial Robustness)
  <!-- - [Citation](#citation) -->


## 2. PERTURBATION-BASED ATTACK

### 2.1 Non-adversarial Perturbation-based Evasion Attack

#### 2.1.1 Current Studies and Underlying Reasons of Vulnerability for CNNs

- [2023/10] **Language Models Hallucinate, but May Excel at Fact Verification** *Jian Guan et al. arXiv.* [[paper](https://arxiv.org/abs/2310.14564)]

- [2023/10] **The Perils & Promises of Fact-checking with Large Language Models** *Dorian Quelle, Alexandre Bovet. arXiv.* [[paper](https://arxiv.org/abs/2310.13549)]

- [2023/10] **Automated Claim Matching with Large Language Models: Empowering Fact-Checkers in the Fight Against Misinformation** *Eun Cheol Choi, Emilio Ferrara. arXiv.* [[paper](https://arxiv.org/abs/2310.09223)]
- [2023/10] **FakeGPT: Fake News Generation, Explanation and Detection of Large Language Models.** *Yue Huang, Lichao Sun. arXiv.* [[paper](https://arxiv.org/abs/2310.05046)]
- [2023/10] **Explainable Claim Verification via Knowledge-Grounded Reasoning with Large Language Models** *Haoran Wang, Kai Shu. arXiv.* [[paper](https://arxiv.org/abs/2310.05253)]
- [2023/09] **Can LLM-Generated Misinformation Be Detected?** *Canyu Chen, Kai Shu. arXiv.* [[paper](https://arxiv.org/abs/2309.13788)]
- [2023/09] **Disinformation Detection: An Evolving Challenge in the Age of LLMs** *Bohan Jiang et al. arXiv.* [[paper](https://arxiv.org/abs/2309.15847)]
- [2023/09] **Can Large Language Models Discern Evidence for Scientific Hypotheses? Case Studies in the Social Sciences.** *Sai Koneru et al. arXiv.* [[paper](https://arxiv.org/abs/2309.06578)]
- [2023/09] **Towards LLM-based Fact Verification on News Claims with a Hierarchical Step-by-Step Prompting Method.** *Xuan Zhang and Wei Gao. AACL 2023.* [[paper](https://arxiv.org/abs/2310.00305)]
- [2023/09] **Analysis of Disinformation and Fake News Detection Using Fine-Tuned Large Language Model.** *Bohdan M. Pavlyshenko. arXiv.* [[paper](https://arxiv.org/abs/2309.04704)]
- [2023/08] **Cheap-fake Detection with LLM using Prompt Engineering.** *Guangyang Wu et al. IEEE ICMEW 2023.* [[paper](https://ieeexplore.ieee.org/document/10221967)]
- [2023/07] **Harnessing the Power of ChatGPT to Decimate Mis/Disinformation: Using ChatGPT for Fake News Detection.** *Kevin Matthe Caramancion. IEEE AIIoT.* [[paper](https://ieeexplore.ieee.org/document/10174450)]
- [2023/07] **Fact-Checking Complex Claims with Program-Guided Reasoning.** *Liangming Pan et al. ACL 2023.* [[paper](https://aclanthology.org/2023.acl-long.386/)]
- [2023/06] **Assessing the Effectiveness of GPT-3 in Detecting False Political Statements: A Case Study on the LIAR Dataset.** *Mars Gokturk Buchholz. arXiv.* [[paper](https://arxiv.org/abs/2306.08190)]
- [2023/06] **A Preliminary Study of ChatGPT on News Recommendation: Personalization, Provider Fairness, Fake News.** *Xinyi Li et al. arXiv.* [[paper](https://arxiv.org/abs/2306.10702)]
- [2023/05] **Self-Checker: Plug-and-Play Modules for Fact-Checking with Large Language Models.** *Miaoran Li et al. arXiv.* [[paper](https://arxiv.org/abs/2305.14623)]
- [2023/05] **Towards Reliable Misinformation Mitigation: Generalization, Uncertainty, and GPT-4.** *Kellin Pelrine et al. arXiv.* [[paper](https://arxiv.org/abs/2305.14928)]
- [2023/04] **Leveraging ChatGPT for Efficient Fact-Checking.** *Emma Hoes et al. psyarxiv.* [[paper](https://osf.io/preprints/psyarxiv/qnjkf/)]
- [2023/04] **Interpretable Unified Language Checking.** *Tianhua Zhang et al. arXiv.* [[paper](https://arxiv.org/abs/2304.03728)]
- [2023/02] **A Multitask, Multi-lingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity.** *Yejin Bang et al. arXiv.* [[paper](https://arxiv.org/abs/2302.04023)]


#### 2.1.2 Attack against Vision Transformers

- [2023/09] **FactLLaMA: Optimizing Instruction-Following Language Models with External Knowledge for Automated Fact-Checking** *Tsun-Hin Cheung and Kin-Man Lam. APSIPA ASC 2023.* [[paper](https://arxiv.org/abs/2309.00240)]
- [2023/07] **FacTool: Factuality Detection in Generative AI -- A Tool Augmented Framework for Multi-Task and Multi-Domain Scenarios** *I-Chun Chern et al. arXiv.* [[paper](https://arxiv.org/abs/2307.13528)]

#### 2.1.3 Comparison between CNNs and ViTs
- [2023/09] **Bad Actor, Good Advisor: Exploring the Role of Large Language Models in Fake News Detection** *Beizhe Hu et al. arXiv.* [[paper](https://arxiv.org/abs/2309.12247)]
- [2023/09] **Detecting Misinformation with LLM-Predicted Credibility Signals and Weak Supervision** *João A. Leite et al. arxiv.* [[paper](https://arxiv.org/abs/2309.07601)]
- [2023/09] **Improving Multiclass Classification of Fake News Using BERT-Based Models and ChatGPT-Augmented Data** *Elena Shushkevich et al. MDPI Inventions.* [[paper](https://www.mdpi.com/2411-5134/8/5/112)]
- [2023/09] **Can Large Language Models Enhance Fake News Detection?: Improving Fake News Detection With Data Augmentation** *Emil Ahlbäck, Max Dougly.* [[paper](https://kth.diva-portal.org/smash/get/diva2:1778369/FULLTEXT01.pdf)]
- [2022/03] **Faking Fake News for Real Fake News Detection: Propaganda-loaded Training Data Generation** *Kung-Hsiang Huang et al. ACL 2023.* [[paper](https://arxiv.org/abs/2203.05386)]
- 
#### 2.1.4 Effectiveness

#### 2.1.5 Imperceptibility


### 2.2 Non-adversarial Perturbation-based Poisoning Attack

- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]
- [2023/04] **Reinforcement Learning-Based Counter-Misinformation Response Generation: A Case Study of COVID-19 Vaccine Misinformation** *Bing He et al. WWW 2023.* [[paper](https://dl.acm.org/doi/10.1145/3543507.3583388)]
- [2023/04] **Working With AI to Persuade: Examining a Large Lan- guage Model’s Ability to Generate Pro-Vaccination Messages** *Elise Karinshak et al. CSCW 2023.* [[paper](https://dl.acm.org/doi/10.1145/3579592)]

#### 2.2.1 Current Studies and Underlying Reasons of Vulnerability
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

### 2.3 Adversarial Perturbation-based Evasion Attack

#### 2.3.1 Basic Generation Methods
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

#### 2.3.2 Underlying Reasons of Vulnerability
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

#### 2.3.3 Black-box Attack
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

##### 2.3.3.1 Black-box Attack
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

##### 2.3.3.2 Query Efficiency
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

##### 2.3.3.3 Transfer-based Attack and Adversarial Transferability (Underlying Reasons)
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

##### 2.3.3.4 Adversarial Transferability Enhancement
###### 2.3.3.4.1 Data augmentation
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]
###### 2.3.3.4.2 Ensemble-based techniques
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]
###### 2.3.3.4.3 Momentum-based methods
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]
###### 2.3.3.4.4 Architecture-oriented methods
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]
###### 2.3.3.4.5 Finding proper substitute models
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]
###### 2.3.3.4.6 Distribution-oriented methods
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]
###### 2.3.3.4.7 Other methods
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

##### 2.3.3.5 Cross-domain/modality Transferability
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]
- 
##### 2.3.3.6 Cross-task Transferability
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

#### 2.3.4 Perturbation against Vision Transformer and Cross-architecture Transferability
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

##### 2.3.4.1 Current Studies and Underlying Reasons for Vulnerability
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

##### 2.3.4.2 Comparison and Cross-architecture Transferability between CNNs and ViTs
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

#### 2.3.5 Non-box Attack
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

#### 2.3.6 Attack against Defense
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

#### 2.3.7 Problem of the Cross-entropy Loss
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

#### 2.3.8 Imperceptibility
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

#### 2.3.9 Diverse Perturbations
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

##### 2.3.9.1 Beyond l_p-norm Perturbations
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

##### 2.3.9.2 Beyond Single or Dual-type Perturbations
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

#### 2.3.10 Unconstrained Perturbations
- [2023/09] **Artificial intelligence is ineffective and potentially harmful for fact checking** *Matthew R. DeVerna et al. arxiv.* [[paper](https://arxiv.org/abs/2308.10800)]

## 3. Adversarial Perturbation-based Poisoning Attack

### 3.1 Targeted Poisoning Attack

- [2023/11] **Adapting Fake News Detection to the Era of Large Language Models.** *Jinyan Su et al. arXiv.* [[paper](https://arxiv.org/abs/2311.04917)]

### 3.2 Backdoor/Trojan Attack

#### 3.2.1 Alleviating Hallucination of LLMs

##### survey

- [2023/09] **Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models.** *Yue Zhang et al. arxiv.* [[paper](https://arxiv.org/abs/2309.01219)]
- [2023/09] **A Survey of Hallucination in Large Foundation Models.** *Vipula Rawte et al. arxiv.* [[paper](https://arxiv.org/abs/2309.05922)]

### 3.3 Untargeted (a.k.a. Availability, Delusive, Indiscriminate) Attack

### 3.4 Transferability
#### 3.4.1 Downstream-agnostic Attack

##### Targeted poisoning
- [2023/09] **Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models.** *Yue Zhang et al. arxiv.* [[paper](https://arxiv.org/abs/2309.01219)]

##### Backdoor attacks
- [2023/09] **Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models.** *Yue Zhang et al. arxiv.* [[paper](https://arxiv.org/abs/2309.01219)]

##### Untargeted attacks
- [2023/09] **Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models.** *Yue Zhang et al. arxiv.* [[paper](https://arxiv.org/abs/2309.01219)]

### 3.5 Imperceptibility

### 3.6 Label-agnostic Attack

### 3.7 Poisoning against Defense

### 3.8 Connection between Evasion Attack and Poisoning Attack

### 3.9 Poisoning against Vision Transformer

### 3.10 Efficiency


## 4 Perturbation-based Augmentation

### 4.1 Defend against Non-adversarial Perturbation-based Evasion Attack

### 4.2 Defense against Adversarial Perturbation-based Evasion Attack

### 4.3 Defense against Non-adversarial Perturbation-based Poisoning Attack

### 4.4 Defense against Adversarial Perturbation-based Poisoning Attack

## 5 Perturbation-based Input Pre-processing

### 5.1 Defend against Non-adversarial Perturbation-based Evasion Attack

### 5.2 Defense against Adversarial Perturbation-based Evasion Attack

### 5.3 Defense against Non-adversarial Perturbation-based Poisoning Attack

### 5.4 Defense against Adversarial Perturbation-based Poisoning Attack


## 6 Randomized Smoothing for Certified Adversarial Robustness

### 6.1 Basic Learning Methods

### 6.2 Diverse Perturbations

#### 6.2.1 Beyond l_2-norm Robustness Certification

#### 6.2.2 Beyond Gaussian Noise

### 6.3 Certified Robustness and its Trade-off with Standard Accuracy

### 6.3 Efficiency

### 6.3 Curse of Dimensionality

### 6.3 Comparison and Connection between Adversarial Training and Randomized Smoothing

### 6.3 Defend against Non-adversarial Perturbation-based Evasion Attack



