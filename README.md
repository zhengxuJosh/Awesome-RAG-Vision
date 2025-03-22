<div align="center">
  <h1>Awesome RAG in Computer Vision</h1>
  <a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome Badge"/></a>
</div>

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/hee9joon/Awesome-Diffusion-Models) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Made With Love](https://img.shields.io/badge/Made%20With-Love-red.svg)](https://github.com/chetanraj/awesome-github-badges)

This repository aims to collect and organize **state-of-the-art papers on Retrieval-Augmented Generation (RAG) in Computer Vision**. RAG has gained significant traction in vision tasks like image understanding, video comprehension, visual generation, and more. By incorporating external retrieval, these approaches can enrich models with additional context, leading to better performance and interpretability.

We encourage researchers who want to showcase their work on **RAG for Vision** to open a Pull Request and add their paper!

---

## Table of Contents

- [Introduction](#introduction)
- [Resources](#resources)
  - [Workshops and Tutorials](#workshops-and-tutorials)
- [Papers](#papers)
  - [Survey and Benchmark](#survey-and-benchmark)
  - [RAG for Vision](#rag-for-vision)
    - [Visual Understanding](#1-visual-understanding)
      - [Image Understanding](#11-image-understanding)
      - [Long Video Understanding](#12-long-video-understanding)
      - [Visual Spatial Understanding](#13-visual-spacial-understanding)
      - [Multi-modal](#14-multi-modal)
      - [Medical Vision](#15-medical-vision)
    - [Visual Generation](#2-visual-generation)
      - [Image (Video) Generation](#21-image-video-generation)
      - [3D Generation](#22-3d-generation)
    - [Embodied AI](#3-embodied-ai)

---
## Introduction

Retrieval-Augmented Generation (RAG) integrates retrieval modules into generative models, allowing them to query external knowledge bases (or memory banks) during inference. In **Computer Vision**, RAG has powered:
- Image captioning and object detection with external knowledge.
- Video QA/comprehension by retrieving context from long transcripts or external references.
- Visual generation with retrieval of reference images, design templates, or domain-specific data.

---

## Resources

### Workshops and Tutorials

## Papers

### Survey and Benchmark

| Year | Paper | Focused Areas    | Main Context                                      | GitHub                                                               |
|------|-------|------------------|---------------------------------------------------|----------------------------------------------------------------------|
| 2023 | Gao *et al.*           | LLMs / NLP       | RAG paradigms and components                        | -                                                                    |
| 2024 | Fan *et al.*           | LLMs / NLP       | RA-LLMs' architectures, training, and applications  | [link](https://advanced-recommender-systems.github.io/RAG-Meets-LLMs/) |
| 2024 | Hu *et al.*            | LLMs / NLP       | RA-LMs' components, evaluation, and limitations     | [link](https://github.com/2471023025/RALM_Survey)                    |
| 2024 | Zhao *et al.*          | LLMs / NLP       | Challenges in data-augmented LLMs                   | -                                                                    |
| 2024 | Gupta *et al.*         | LLMs / NLP       | Advancements and downstream tasks of RAG            | -                                                                    |
| 2024 | Zhao *et al.*          | RAG in AIGC      | RAG applications across modalities                  | [link](https://github.com/PKU-DAIR/RAG-Survey)                       |
| 2024 | Yu *et al.*            | LLMs / NLP       | Unified evaluation process of RAG                   | [link](https://github.com/YHPeter/Awesome-RAG-Evaluation)            |
| 2024 | Procko *et al.*        | Graph Learning   | Knowledge graphs with LLM RAG                       | -                                                                    |
| 2024 | Zhou *et al.*          | Trustworthiness AI | Six dimensions and benchmarks about Trustworthy RAG | [link](https://github.com/smallporridge/TrustworthyRAG)             |
| 2025 | Singh *et al.*         | AI Agent         | Participles and evaluation                          | [link](https://github.com/asinghcsu/AgenticRAG-Survey)               |
| 2025 | Ni *et al.*            | Trustworthiness AI | Road-map and discussion                             | [link](https://github.com/Arstanley/Awesome-Trustworthy-Retrieval-Augmented-Generation) |
| 2025 | **_Ours_**             | **_Computer Vision_** | **_RAG for visual understanding and generation_**   | [link](https://github.com/zhengxuJosh/Awesome-RAG-Vision)            |

## RAG for Vision

### 1 Visual Understanding

#### 1.1 Image Understanding
| Title & Link                                                                                                           | Authors        | Venue/Date         |
|------------------------------------------------------------------------------------------------------------------------|----------------|--------------------|
| [**DIR: Retrieval-Augmented Image Captioning with Comprehensive Understanding**](https://arxiv.org/pdf/2412.01115)     | Wu *et al.*    | Arxiv 2024 (Dec)   |
| [**Retrieval-Augmented Open-Vocabulary Object Detection**](https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_Retrieval-Augmented_Open-Vocabulary_Object_Detection_CVPR_2024_paper.pdf) | Kim *et al.*   | CVPR 2024          |
| [**Understanding Retrieval Robustness for Retrieval-Augmented Image Captioning**](https://arxiv.org/pdf/2406.02265)    | Li *et al.*    | Arxiv 2024 (Aug)   |
| [**Retrieval-Augmented Classification for Long-Tail Visual Recognition**](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Learning_Customized_Visual_Models_With_Retrieval-Augmented_Knowledge_CVPR_2023_paper.pdf) | Long *et al.*  | CVPR 2022          |
| [**Learning Customized Visual Models with Retrieval-Augmented Knowledge**](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Learning_Customized_Visual_Models_With_Retrieval-Augmented_Knowledge_CVPR_2023_paper.pdf) | Liu *et al.*   | CVPR 2023          |
<!--
DIR: Hao Wu, Zhihang Zhong, Xiao Sun  
Retrieval-Augmented OVD: Jooyeon Kim, Eulrang Cho, Sehyung Kim, Hyunwoo J. Kim  
Understanding Retrieval Robustness: Wenyan Li, Jiaang Li, Rita Ramos, Raphael Tang, Desmond Elliott  
Retrieval-Augmented Classification: Alexander Long, Wei Yin, Thalaiyasingam Ajanthan, Vu Nguyen, Pulak Purkait, Ravi Garg, Alan Blair, Chunhua Shen, Anton van den Hengel  
Learning Customized Visual Models: Haotian Liu, Kilho Son, Jianwei Yang, Ce Liu, Jianfeng Gao, Yong Jae Lee, Chunyuan Li  
-->


#### 1.2 (Long) Video Understanding
| Title & Link                                                                                                           | Authors        | Venue/Date         |
|------------------------------------------------------------------------------------------------------------------------|----------------|--------------------|
| [**Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension**](https://arxiv.org/pdf/2411.13093)      | Luo *et al.*   | Arxiv 2024 (Nov)   |
| [**ViTA: An Efficient Video-to-Text Algorithm using VLM for RAG-based Video Analysis System**](https://aclanthology.org/2024.emnlp-main.62.pdf) | Arefeen *et al.* | CVPRW 2024         |
| [**iRAG: Advancing RAG for Videos with an Incremental Approach**](https://dl.acm.org/doi/pdf/10.1145/3627673.3680088?casa_token=CDXIXZP0y9QAAAAA:obaFKtQODdGsI3pB22GWuGH2dODwF7N0dj1dl58WfSwavmvrp_1eeaHXj6c2XCQyt-9vF1r1QrUd) | Arefeen *et al.* | CIKM 2024          |
<!--
Video-RAG: Yongdong Luo, Xiawu Zheng, Xiao Yang, Guilin Li, Haojia Lin, Jinfa Huang, Jiayi Ji, Fei Chao, Jiebo Luo, Rongrong Ji  
ViTA: Md Adnan Arefeen, Biplob Debnath, Md Yusuf Sarwar Uddin, Srimat Chakradhar  
iRAG: Md Adnan Arefeen, Md Yusuf Sarwar Uddin, Biplob Debnath, Srimat Chakradhar  
-->


#### 1.3 Visual Spacial Understanding

| Title & Link                                                                                                                                                      | Authors       | Venue/Date   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|--------------|
| [**RAG-Guided Large Language Models for Visual Spatial Description with Adaptive Hallucination Corrector**](https://dl.acm.org/doi/abs/10.1145/3664647.3688990?casa_token=SlLR5jgRRkgAAAAA:DzC124tFMWQSMYkKRGkPTwU-aaT7TSv_iVjE-dsZtbna9j3zCYX1A6qcfgmpEKTms8DoZDgplc5u8g) | Yu *et al.*   | ACM MM 2024  |
<!--
RAG-Guided Large Language Models for Visual Spatial Description with Adaptive Hallucination Corrector:  
Jun Yu, Yunxiang Zhang, Zerui Zhang, Zhao Yang, Gongpeng Zhao, Fengzhao Sun, Fanrui Zhang, Qingsong Liu, Jianqing Sun, Jiaen Liang, Yaohui Zhang
-->


#### 1.4 Multi-modal

| Title & Link                                                                                                                                      | Authors         | Venue/Date         |
|---------------------------------------------------------------------------------------------------------------------------------------------------|------------------|--------------------|
| [**mR2AG: Multimodal Retrieval-Reflection-Augmented Generation for Knowledge-Based VQA**](https://arxiv.org/pdf/2411.15041)                      | Zhang *et al.*   | Arxiv 2024 (Nov)   |
| [**Wiki-LLaVA: Hierarchical Retrieval-Augmented Generation for Multimodal LLMs**](https://openaccess.thecvf.com/content/CVPR2024W/MMFM/papers/Caffagni_Wiki-LLaVA_Hierarchical_Retrieval-Augmented_Generation_for_Multimodal_LLMs_CVPRW_2024_paper.pdf) | Caffagni *et al.*| CVPRW 2024         |
| [**M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding**](https://arxiv.org/pdf/2410.21943)             | Cho *et al.*     | Arxiv 2024 (Nov)   |
| [**UniRAG: Universal Retrieval Augmentation for Multi-Modal Large Language Models**](https://arxiv.org/pdf/2405.10311)                           | Sharifymoghaddam *et al.* | Arxiv 2024 (Oct)   |
| [**MRAG-Bench: Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models**](https://arxiv.org/pdf/2410.08182)                          | Hu *et al.*      | ICLR 2025 (Oct)    |
| [**VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents**](https://arxiv.org/pdf/2410.10594)                          | Yu *et al.*      | ICLR 2025 (Oct)    |
| [**RoRA-VLM: Robust Retrieval Augmentation for Vision Language Models**](https://arxiv.org/pdf/2410.08876)                                       | Qi *et al.*      | Arxiv 2024 (Oct)   |
| [**Beyond Text: Optimizing RAG with Multimodal Inputs for Industrial Applications**](https://arxiv.org/pdf/2410.21943)                           | Riedler *et al.* | Arxiv 2024 (Oct)   |
| [**SURf: Teaching Large Vision-Language Models to Selectively Utilize Retrieved Information**](https://aclanthology.org/2024.emnlp-main.434.pdf) | Sun *et al.*     | EMNLP 2024 (Sep)   |
| [**ColPali: Efficient Document Retrieval with Vision Language Models**](https://openreview.net/pdf?id=ogjBpZ8uSi)                                | Faysse *et al.*  | ICLR 2025 (Jul)    |
| [**MLLM Is a Strong Reranker**](https://arxiv.org/pdf/2409.14083)                                                                                | Chen *et al.*    | Arxiv 2024 (Jul)   |
| [**RAVEN: Multitask Retrieval Augmented Vision-Language Learning**](https://openreview.net/pdf?id=GMalvQu0XL)                                    | Rao *et al.*     | COLM 2024 (Jun)    |
| [**SearchLVLMs**](https://papers.nips.cc/paper_files/paper/2024/file/76954b4a44e158e738b4c64494977c6a-Paper-Conference.pdf)                      | Li *et al.*      | NIPS 2024 (May)    |
| [**UDKAG**](https://arxiv.org/abs/2405.14554v1)                                                                                                   | Li *et al.*      | CoRR 2024 (May)    |
| [**Retrieval Meets Reasoning**](https://arxiv.org/pdf/2409.14083)                                                                                | Tan *et al.*     | Arxiv 2024 (Apr)   |
| [**RAR**](https://arxiv.org/pdf/2409.14083)                                                                                                       | Liu *et al.*     | Arxiv 2024 (Mar)   |
| [**Fine-grained Late-interaction Multi-modal Retrieval for RAG-VQA**](https://papers.nips.cc/paper_files/paper/2023/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf) | Lin *et al.*     | NIPS 2023 (Oct)    |
| [**Retrieval-based Knowledge Augmented Vision Language Pre-training**](https://arxiv.org/pdf/2304.13923)                                         | Rao *et al.*     | ACMMM 2023 (Apr)   |
| [**ReVeaL**](https://arxiv.org/pdf/2212.05221)                                                                                                    | Hu *et al.*      | CVPR 2023 (Apr)    |
| [**Murag**](https://aclanthology.org/2022.emnlp-main.375.pdf)                                                                                    | Chen *et al.*    | EMNLP 2022 (Oct)   |
<!--
mR2AG: Tao Zhang, Ziqi Zhang, Zongyang Ma, Yuxin Chen, Zhongang Qi, Chunfeng Yuan, Bing Li, Junfu Pu, Yuxuan Zhao, Zehua Xie, Jin Ma, Ying Shan, Weiming Hu  
Wiki-LLaVA: Davide Caffagni, Federico Cocchi, Nicholas Moratelli, Sara Sarto, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara  
M3DocRAG: Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, Mohit Bansal  
UniRAG: Sahel Sharifymoghaddam, Shivani Upadhyay, Wenhu Chen, Jimmy Lin  
MRAG-Bench: Wenbo Hu, Jia-Chen Gu, Zi-Yi Dou, Mohsen Fayyaz, Pan Lu, Kai-Wei Chang, Nanyun Peng  
VisRAG: Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, Maosong Sun  
RoRA-VLM: Jingyuan Qi, Zhiyang Xu, Rulin Shao, Yang Chen, Jin Di, Yu Cheng, Qifan Wang, Lifu Huang  
Beyond Text: Monica Riedler, Stefan Langer  
SURf: Jiashuo Sun, Jihai Zhang, Yucheng Zhou, Zhaochen Su, Xiaoye Qu, Yu Cheng  
ColPali: Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, Pierre Colombo  
MLLM Is a Strong Reranker: Zhanpeng Chen, Chengjin Xu, Yiyan Qi, Jian Guo  
RAVEN: Varun Nagaraj Rao, Siddharth Choudhary, Aditya Deshpande, Ravi Kumar Satzoda, Srikar Appalaraju  
SearchLVLMs: Chuanhao Li, Zhen Li, Chenchen Jing, Shuo Liu, Wenqi Shao, Yuwei Wu, Ping Luo, Yu Qiao, Kaipeng Zhang  
UDKAG: Chuanhao Li, Zhen Li, Chenchen Jing, Shuo Liu, Wenqi Shao, Yuwei Wu, Ping Luo, Yu Qiao, Kaipeng Zhang  
Retrieval Meets Reasoning: Cheng Tan, Jingxuan Wei, Linzhuang Sun, Zhangyang Gao, Siyuan Li, Bihui Yu, Ruifeng Guo, Stan Z. Li  
RAR: Ziyu Liu, Zeyi Sun, Yuhang Zang, Wei Li, Pan Zhang, Xiaoyi Dong, Yuanjun Xiong, Dahua Lin, Jiaqi Wang  
Fine-grained Retrieval: Weizhe Lin, Jinghong Chen, Jingbiao Mei, Alexandru Coca, Bill Byrne  
Retrieval-based Pre-training: Jiahua Rao, Zifei Shan, Longpo Liu, Yao Zhou, Yuedong Yang  
ReVeaL: Ziniu Hu, Ahmet Iscen, Chen Sun, Zirui Wang, Kai-Wei Chang, Yizhou Sun, Cordelia Schmid, David A. Ross, Alireza Fathi  
Murag: Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, William W. Cohen  
-->

#### 1.5 Medical Vision
| Title & Link                                                                                                                                      | Authors       | Venue/Date       |
|---------------------------------------------------------------------------------------------------------------------------------------------------|---------------|------------------|
| [**Mmed-rag: Versatile multimodal rag system for medical vision language models**](https://arxiv.org/pdf/2410.13085)                             | Xia *et al.*  | Arxiv 2024 (Oct) |
| [**Rule: Reliable multimodal rag for factuality in medical vision language models**](https://aclanthology.org/2024.emnlp-main.62.pdf)            | Xia *et al.*  | EMNLP 2024       |
<!--
Mmed-rag: Peng Xia, Kangyu Zhu, Haoran Li, Tianze Wang, Weijia Shi, Sheng Wang, Linjun Zhang, James Zou, Huaxiu Yao  
Rule: Peng Xia, Kangyu Zhu, Haoran Li, Hongtu Zhu, Yun Li, Gang Li, Linjun Zhang, Huaxiu Yao  
-->

### 2 Visual Generation 

#### 2.1 Image (Video) Generation

| Title & Link                                                                                                                                      | Authors       | Venue/Date       |
|---------------------------------------------------------------------------------------------------------------------------------------------------|---------------|------------------|
| [**FairRAG: Fair Human Generation via Fair Retrieval Augmentation**](https://openaccess.thecvf.com/content/CVPR2024/papers/Shrestha_FairRAG_Fair_Human_Generation_via_Fair_Retrieval_Augmentation_CVPR_2024_paper.pdf) | Shrestha *et al.* | CVPR 2024        |
| [**GarmentAligner (ECCV)**](https://link.springer.com/chapter/10.1007/978-3-031-72698-9_9)                                                       | Zhang *et al.* | ECCV 2025        |
| [**Retrieval-Augmented Diffusion Models**](https://proceedings.neurips.cc/paper_files/paper/2022/file/62868cc2fc1eb5cdf321d05b4b88510c-Paper-Conference.pdf) | Blattmann *et al.* | NIPS 2022        |
| [**Label-Retrieval-Augmented Diffusion Models**](https://proceedings.neurips.cc/paper_files/paper/2023/file/d191ba4c8923ed8fd8935b7c98658b5f-Paper-Conference.pdf) | Chen *et al.* | NIPS 2023        |
| [**CPR: Retrieval Augmented Generation for Copyright Protection**](https://openaccess.thecvf.com/content/CVPR2024/papers/Golatkar_CPR_Retrieval_Augmented_Generation_for_Copyright_Protection_CVPR_2024_paper.pdf) | Golatkar *et al.* | CVPR 2023        |
| [**BrainRAM**](https://dl.acm.org/doi/pdf/10.1145/3664647.3681296)                                                                               | Xie *et al.*     | MM 2024          |
| [**Animate-A-Story**](https://arxiv.org/pdf/2307.06940)                                                                                          | He *et al.*      | Arxiv 2023       |
| [**RealGen**](https://arxiv.org/pdf/2312.13303)                                                                                                   | Ding *et al.*    | ECCV 2024        |
| [**Grounding Language Models for Visual Entity Recognition**](https://arxiv.org/pdf/2402.18695)                                                  | Xiao *et al.*    | ECCV 2024        |
| [**GarmentAligner (Arxiv)**](https://arxiv.org/pdf/2408.12352)                                                                                   | Zhang *et al.*   | ECCV 2024        |
| [**Retrieval-Augmented Layout Transformer**](https://openaccess.thecvf.com/content/CVPR2024/papers/Horita_Retrieval-Augmented_Layout_Transformer_for_Content-Aware_Layout_Generation_CVPR_2024_paper.pdf) | Horita *et al.* | CVPR 2024        |
| [**The Neglected Tails in Vision-Language Models**](https://openaccess.thecvf.com/content/CVPR2024/papers/Horita_Retrieval-Augmented_Layout_Transformer_for_Content-Aware_Layout_Generation_CVPR_2024_paper.pdf) | Parashar *et al.* | CVPR 2024       |
| [**Prompt Expansion for Adaptive Text-to-Image Generation**](https://arxiv.org/pdf/2312.16720)                                                   | Datta *et al.*   | ACL 2024         |
| [**Factuality Tax of Diversity-Intervened Generation**](https://arxiv.org/pdf/2407.00377)                                                        | Wan *et al.*     | EMNLP 2024       |
| [**Diffusion-Based Augmentation for Captioning and Retrieval**](https://openaccess.thecvf.com/content/ICCV2023W/e-Heritage/papers/Cioni_Diffusion_Based_Augmentation_for_Captioning_and_Retrieval_in_Cultural_Heritage_ICCVW_2023_paper.pdf) | Cioni *et al.* | ICCV 2023        |
| [**ReMoDiffuse**](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_ReMoDiffuse_Retrieval-Augmented_Motion_Diffusion_Model_ICCV_2023_paper.pdf) | Zhang *et al.* | ICCV 2023        |
| [**Re-imagen**](https://arxiv.org/pdf/2209.14491)                                                                                                 | Chen *et al.*    | Arxiv 2022       |
| [**Instruct-Imagen**](https://openaccess.thecvf.com/content/CVPR2024/papers/Hu_Instruct-Imagen_Image_Generation_with_Multi-modal_Instruction_CVPR_2024_paper.pdf) | Hu *et al.* | CVPR 2024        |
| [**ImageRAG**](https://arxiv.org/pdf/2502.09411)                                                                                                  | Shalev-Arkushin *et al.* | Arxiv 2025  |
| [**FineRAG**](https://aclanthology.org/2025.coling-main.741.pdf)                                                                                 | Yuan *et al.*    | COLING 2025      |
| [**RealRAG**](https://arxiv.org/pdf/2502.00848)                                                                                                   | Lyu *et al.*     | Arxiv 2025       |
<!--
FairRAG: Robik Shrestha, Yang Zou, Qiuyu Chen, Zhiheng Li, Yusheng Xie, Siqi Deng  
GarmentAligner (ECCV): Shiyue Zhang, Zheng Chong, Xujie Zhang, Hanhui Li, Yuhao Cheng, Yiqiang Yan & Xiaodan Liang  
Retrieval-Augmented Diffusion Models: Andreas Blattmann, Robin Rombach, Kaan Oktay, Jonas Müller, Björn Ommer  
Label-Retrieval-Augmented: Jian Chen, Ruiyi Zhang, Tong Yu, Rohan Sharma, Zhiqiang Xu, Tong Sun, Changyou Chen  
CPR: Aditya Golatkar, Alessandro Achille, Luca Zancato, Yu-Xiang Wang, Ashwin Swaminathan, Stefano Soatto  
BrainRAM: Dian Xie, Peiang Zhao, Jiarui Zhang, Kangqi Wei, Xiaobao Ni, Jiong Xia  
Animate-A-Story: He Yingqing, Xia Menghan, Chen Haoxin, Cun Xiaodong, Gong Yuan, Xing Jinbo, Zhang Yong, Wang Xintao, Weng Chao, Shan Ying, Chen Qifeng  
RealGen: Wenhao Ding, Yulong Cao, Ding Zhao, Chaowei Xiao, Marco Pavone  
Grounding VLMs: Zilin Xiao, Ming Gong, Paola Cascante-Bonilla, Xingyao Zhang, Jie Wu, Vicente Ordonez  
GarmentAligner (Arxiv): Shiyue Zhang, Zheng Chong, Xujie Zhang, Hanhui Li, Yuhao Cheng, Yiqiang Yan, Xiaodan Liang  
Layout Transformer: Daichi Horita, Naoto Inoue, Kotaro Kikuchi, Kota Yamaguchi, Kiyoharu Aizawa  
Neglected Tails: Shubham Parashar, Zhiqiu Lin, Tian Liu, Xiangjue Dong, Yanan Li, Deva Ramanan, James Caverlee, Shu Kong  
Prompt Expansion: Siddhartha Datta, Alexander Ku, Deepak Ramachandran, Peter Anderson  
Factuality Tax: Yixin Wan, Di Wu, Haoran Wang, Kai-Wei Chang  
Diffusion for Heritage: Dario Cioni, Lorenzo Berlincioni, Federico Becattini, Alberto del Bimbo  
ReMoDiffuse: Mingyuan Zhang, Xinying Guo, Liang Pan, Zhongang Cai, Fangzhou Hong, Huirong Li, Lei Yang, Ziwei Liu  
Re-imagen: Wenhu Chen, Hexiang Hu, Chitwan Saharia, William W. Cohen  
Instruct-Imagen: Hexiang Hu, Kelvin C.K. Chan, Yu-Chuan Su, Wenhu Chen, Yandong Li, Kihyuk Sohn, Yang Zhao, Xue Ben, Boqing Gong, William Cohen, Ming-Wei Chang, Xuhui Jia  
ImageRAG: Rotem Shalev-Arkushin, Rinon Gal, Amit H. Bermano, Ohad Fried  
FineRAG: Huaying Yuan, Ziliang Zhao, Shuting Wang, Shitao Xiao, Minheng Ni, Zheng Liu, Zhicheng Dou  
RealRAG: Yuanhuiyi Lyu, Xu Zheng, Lutao Jiang, Yibo Yan, Xin Zou, Huiyu Zhou, Linfeng Zhang, Xuming Hu  
-->


#### 2.2 3D Generation
| Title & Link                                                                                                                                      | Authors         | Venue/Date       |
|---------------------------------------------------------------------------------------------------------------------------------------------------|------------------|------------------|
| [**Phidias: A Generative Model for Creating 3D Content from Text, Image, and 3D Conditions**](https://arxiv.org/pdf/2409.11406)                   | Wang *et al.*    | Arxiv 2024 (Sep) |
| [**Retrieval-Augmented Score Distillation for Text-to-3D Generation**](https://arxiv.org/pdf/2402.02972)                                         | Seo *et al.*     | ICML 2024        |
| [**Diorama: Unleashing Zero-shot Single-view 3D Scene Modeling**](https://arxiv.org/pdf/2411.19492)                                              | Wu *et al.*      | Arxiv 2024 (Nov) |
| [**Interaction-based Retrieval-augmented Diffusion for Protein 3D Generation**](https://openreview.net/pdf?id=eejhD9FCP3)                         | Huang *et al.*   | ICML 2024        |
| [**ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model**](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_ReMoDiffuse_Retrieval-Augmented_Motion_Diffusion_Model_ICCV_2023_paper.pdf) | Zhang *et al.*   | ICCV 2023        |
<!--
Phidias: Zhenwei Wang, Tengfei Wang, Zexin He, Gerhard Hancke, Ziwei Liu, Rynson W.H. Lau  
Retrieval-Augmented Score Distillation: Junyoung Seo, Susung Hong, Wooseok Jang, Ines Hyeonsu Kim, Minseop Kwak, Doyup Lee, Seungryong Kim  
Diorama: Qirui Wu, Denys Iliash, Daniel Ritchie, Manolis Savva, Angel X. Chang  
Interaction-based Diffusion: Zhilin Huang, Ling Yang, Xiangxin Zhou, Chujun Qin, Yijie Yu, Xiawu Zheng, Zikun Zhou, Wentao Zhang, Yu Wang, Wenming Yang  
ReMoDiffuse: Mingyuan Zhang, Xinying Guo, Liang Pan, Zhongang Cai, Fangzhou Hong, Huirong Li, Lei Yang, Ziwei Liu  
-->


### 3. Embodied AI
| Title & Link                                                                                                                                      | Authors        | Venue/Date       |
|---------------------------------------------------------------------------------------------------------------------------------------------------|----------------|------------------|
| [**P-RAG: Progressive Retrieval Augmented Generation For Planning on Embodied Everyday Task**](https://dl.acm.org/doi/pdf/10.1145/3664647.3680661) | Xu *et al.*     | ACM MM 2024      |
| [**Realgen: Retrieval augmented generation for controllable traffic scenarios**](https://arxiv.org/pdf/2312.13303)                                | Ding *et al.*   | ECCV 2024        |
| [**Retrieval-Augmented Embodied Agents**](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_Retrieval-Augmented_Embodied_Agents_CVPR_2024_paper.pdf) | Zhu *et al.*     | CVPR 2024        |
| [**ENWAR: A RAG-empowered Multi-Modal LLM Framework for Wireless Environment Perception**](https://arxiv.org/pdf/2410.18104)                      | Nazar *et al.*  | Arxiv 2024 (Oct) |
| [**Embodied-RAG: General Non-parametric Embodied Memory for Retrieval and Generation**](https://arxiv.org/pdf/2409.18313)                         | Xie *et al.*    | Arxiv 2024 (Oct) |
| [**RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning**](https://arxiv.org/abs/2402.10828)               | Yuan *et al.*   | Arxiv 2024 (May) |
<!--
P-RAG: Weiye Xu, Min Wang, Wengang Zhou, Houqiang Li  
Realgen: Wenhao Ding, Yulong Cao, Ding Zhao, Chaowei Xiao, Marco Pavone  
Retrieval-Augmented Embodied Agents: Yichen Zhu, Zhicai Ou, Xiaofeng Mou, Jian Tang  
ENWAR: Ahmad M. Nazar, Abdulkadir Celik, Mohamed Y. Selim, Asmaa Abdallah, Daji Qiao, Ahmed M. Eltawil  
Embodied-RAG: Quanting Xie, So Yeon Min, Tianyi Zhang, Kedi Xu, Aarav Bajaj, Ruslan Salakhutdinov, Matthew Johnson-Roberson, Yonatan Bisk  
RAG-Driver: Jianhao Yuan, Shuyang Sun, Daniel Omeiza, Bo Zhao, Paul Newman, Lars Kunze, Matthew Gadd  
-->

