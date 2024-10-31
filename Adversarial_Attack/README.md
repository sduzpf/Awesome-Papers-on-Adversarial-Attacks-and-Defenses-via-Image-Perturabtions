## Table of Content (ToC)

- [image-perturbations-survey](#image-perturbations-survey)
  - [2. PERTURBATION-BASED ATTACK](#2-PERTURBATION-BASED ATTACK)
    - [2.1 Non-adversarial Perturbation-based Evasion Attack](#21-Non-adversarial Perturbation-based Evasion Attack)
      - [2.1.1 Current Studies and Underlying Reasons of Vulnerability for CNNs](#211-Current Studies for CNNs)
      - [2.1.2 Attack against Vision Transformers](#212-ttack against Vision Transformers)
      - [2.1.3 Comparison between CNNs and ViTs](#213-Comparison between CNNs and ViTs)
      - [2.1.4 Effectiveness](#213-Effectiveness)
      - [2.1.5 Imperceptibility](#215-Imperceptibility)
    - [2.2 on-adversarial Perturbation-based Poisoning Attack](#22-on-adversarial Perturbation-based Poisoning Attack)
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
  <!-- - [Citation](#citation) -->

## 2. PERTURBATION-BASED ATTACK

### 2.1 Non-adversarial Perturbation-based Evasion Attack
#### 2.1.1 Current Studies and Underlying Reasons of Vulnerability for CNNs
#### 2.1.2 Attack against Vision Transformers
#### 2.1.3 Comparison between CNNs and ViTs
#### 2.1.4 Effectiveness
#### 2.1.5 Imperceptibility



