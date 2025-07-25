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

#### **RAG for Image**
- [Multimodal RAG using Langchain Expression Language And GPT4-Vision](https://medium.aiplanet.com/multimodal-rag-using-langchain-expression-language-and-gpt4-vision-8a94c8b02d21)
- [A Comprehensive Guide to Building Multimodal RAG Systems](https://www.analyticsvidhya.com/blog/2024/09/guide-to-building-multimodal-rag-systems/)
- [Guide to Multimodal RAG for Images and Text (in 2025)](https://medium.com/kx-systems/guide-to-multimodal-rag-for-images-and-text-10dab36e3117)
- [Building an Image Search RAG App with Llama 3.2 Vision](https://blog.stackademic.com/building-an-image-search-rag-app-with-llama-3-2-vision-a-step-by-step-implementation-guide-d2f79c1c4c15)
- [Improve Your Stable Diffusion Prompts with Retrieval-Augmented Generation](https://aws.amazon.com/cn/blogs/machine-learning/improve-your-stable-diffusion-prompts-with-retrieval-augmented-generation/)


#### **RAG for video**
- [Building Multimodal RAG Application for Video Preprocessing](https://pub.towardsai.net/building-multimodal-rag-application-4-video-preprocessing-multimodal-rag-abf086f81221)
- [Multimodal RAG Chat with Videos and the Future of AI Interaction](https://ai.plainenglish.io/multimodal-rag-chat-with-videos-and-the-future-of-ai-interaction-e427b755689c)
- [Multimodal RAG for Advanced Video Processing with LlamaIndex and LanceDB](https://www.llamaindex.ai/blog/multimodal-rag-for-advanced-video-processing-with-llamaindex-lancedb-33be4804822e)
- [Multimodal RAG for processing videos using OpenAI GPT4V and LanceDB vectorstore](https://docs.llamaindex.ai/en/stable/examples/multi_modal/multi_modal_video_RAG/)
- [RAG (Q/A) of Videos with LLM](https://www.kaggle.com/code/gabrielvinicius/rag-q-a-of-videos-with-llm)
- [An Easy Introduction to Multimodal Retrieval-Augmented Generation for Video and Audio](https://developer.nvidia.com/blog/an-easy-introduction-to-multimodal-retrieval-augmented-generation-for-video-and-audio/)

#### **RAG for Document**
- [Multimodal RAG for PDFs with Text, Images, and Charts](https://pathway.com/developers/templates/multimodal-rag/)
- [Multimodal Retrieval-Augmented Generation (RAG) with Document Retrieval (ColPali) and Vision Language Models (VLMs)](https://huggingface.co/learn/cookbook/multimodal_rag_using_document_retrieval_and_vlms)
- [Multi-Vector Retriever for RAG on tables, text, and images](https://blog.langchain.dev/semi-structured-multi-modal-rag/)
- [Build an AI-powered multimodal RAG system with Docling and Granite](https://www.ibm.com/think/tutorials/build-multimodal-rag-langchain-with-docling-granite)

#### **Other Related Resources**
- [Multimodal Retrieval Augmented Generation(RAG)](https://weaviate.io/blog/multimodal-rag)
- [Tutorial | Build a multimodal knowledge bank for a RAG project](https://knowledge.dataiku.com/latest/gen-ai/rag/tutorial-multimodal-embedding.html)

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
| Title                                                                                             | Authors       | Venue/Date       | Paper Link                                                                                       |
|---------------------------------------------------------------------------------------------------|---------------|------------------|--------------------------------------------------------------------------------------------------|
| DIR: Retrieval-Augmented Image Captioning with Comprehensive Understanding                        | Wu *et al.*   | Arxiv 2024 (Dec) | [paper](https://arxiv.org/pdf/2412.01115)                                                       |
| Retrieval-Augmented Open-Vocabulary Object Detection                                               | Kim *et al.*  | CVPR 2024        | [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_Retrieval-Augmented_Open-Vocabulary_Object_Detection_CVPR_2024_paper.pdf) |
| Understanding Retrieval Robustness for Retrieval-Augmented Image Captioning                       | Li *et al.*   | Arxiv 2024 (Aug) | [paper](https://arxiv.org/pdf/2406.02265)                                                       |
| Retrieval-Augmented Classification for Long-Tail Visual Recognition                               | Long *et al.* | CVPR 2022        | [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Learning_Customized_Visual_Models_With_Retrieval-Augmented_Knowledge_CVPR_2023_paper.pdf) |
| Learning Customized Visual Models with Retrieval-Augmented Knowledge                              | Liu *et al.*  | CVPR 2023        | [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Learning_Customized_Visual_Models_With_Retrieval-Augmented_Knowledge_CVPR_2023_paper.pdf) |

<!--
DIR: Hao Wu, Zhihang Zhong, Xiao Sun  
Retrieval-Augmented OVD: Jooyeon Kim, Eulrang Cho, Sehyung Kim, Hyunwoo J. Kim  
Understanding Retrieval Robustness: Wenyan Li, Jiaang Li, Rita Ramos, Raphael Tang, Desmond Elliott  
Retrieval-Augmented Classification: Alexander Long, Wei Yin, Thalaiyasingam Ajanthan, Vu Nguyen, Pulak Purkait, Ravi Garg, Alan Blair, Chunhua Shen, Anton van den Hengel  
Learning Customized Visual Models: Haotian Liu, Kilho Son, Jianwei Yang, Ce Liu, Jianfeng Gao, Yong Jae Lee, Chunyuan Li  
-->


#### 1.2 (Long) Video Understanding
| Title                                                                                             | Authors         | Venue/Date        | Paper Link                                                                                                                                                                                                                         |
|---------------------------------------------------------------------------------------------------|------------------|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Multi-RAG: A Multimodal Retrieval-Augmented Generation System for Adaptive Video Understanding         | Mao *et al.*   | Arxiv 2025 (Jun)   | [paper](https://arxiv.org/pdf/2505.23990) | 
| VRAG: Retrieval-Augmented Video Question Answering for Long-Form Videos          | Gia *et al.*   | CVPRW 2025  | [paper](https://openaccess.thecvf.com/content/CVPR2025W/IViSE/papers/Gia_VRAG_Retrieval-Augmented_Video_Question_Answering_for_Long-Form_Videos_CVPRW_2025_paper.pdf) |
| Streaming Video Understanding and Multi-round Interaction with Memory-enhanced Knowledge          | Xiong *et al.*   | ICLR 2025        | [paper](https://arxiv.org/abs/2501.13468) |
| Temporal Preference Optimization for Long-Form Video Understanding                                | Li *et al.*      | Arxiv 2025 (Jan) | [paper](https://arxiv.org/abs/2501.13919) |
| StreamingRAG: Real-time Contextual Retrieval and Generation Framework                            | Sankaradas *et al.* | Arxiv 2025 (Jan) | [paper](https://arxiv.org/abs/2501.14101) |
| VideoAuteur: Towards Long Narrative Video Generation                                             | Xiao *et al.*    | Arxiv 2025 (Jan) | [paper](https://arxiv.org/abs/2501.06173) |
| FrameFusion: Combining Similarity and Importance for Video Token Reduction on Large Visual Language Models | Fu *et al.*   | Arxiv 2024 (Dec) | [paper](https://arxiv.org/abs/2501.01986) |
| Vinci: A Real-time Embodied Smart Assistant based on Egocentric Vision-Language Model           | Huang *et al.*   | Arxiv 2024 (Dec) | [paper](https://arxiv.org/abs/2412.21080) |
| Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models                | Yi *et al.*      | Arxiv 2024 (Dec) | [paper](https://arxiv.org/abs/2412.18609) |
| Goldfish: Vision-Language Understanding of Arbitrarily Long Videos                               | Ataallah *et al.*| Arxiv 2024 (Jul) | [paper](https://arxiv.org/abs/2407.12679) |
| Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension                         | Luo *et al.*     | Arxiv 2024 (Nov) | [paper](https://arxiv.org/pdf/2411.13093) |
| ViTA: An Efficient Video-to-Text Algorithm using VLM for RAG-based Video Analysis System         | Arefeen *et al.* | CVPRW 2024       | [paper](https://aclanthology.org/2024.emnlp-main.62.pdf) |
| iRAG: Advancing RAG for Videos with an Incremental Approach                                       | Arefeen *et al.* | CIKM 2024        | [paper](https://dl.acm.org/doi/pdf/10.1145/3627673.3680088?casa_token=CDXIXZP0y9QAAAAA:obaFKtQODdGsI3pB22GWuGH2dODwF7N0dj1dl58WfSwavmvrp_1eeaHXj6c2XCQyt-9vF1r1QrUd) |
<!--
Video-RAG: Yongdong Luo, Xiawu Zheng, Xiao Yang, Guilin Li, Haojia Lin, Jinfa Huang, Jiayi Ji, Fei Chao, Jiebo Luo, Rongrong Ji  
ViTA: Md Adnan Arefeen, Biplob Debnath, Md Yusuf Sarwar Uddin, Srimat Chakradhar  
iRAG: Md Adnan Arefeen, Md Yusuf Sarwar Uddin, Biplob Debnath, Srimat Chakradhar  
-->


#### 1.3 Visual Spacial Understanding

| Title                                                                                             | Authors      | Venue/Date    | Paper Link                                                                                                                                                                                                                                                                                             |
|---------------------------------------------------------------------------------------------------|--------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| RAG-Guided Large Language Models for Visual Spatial Description with Adaptive Hallucination Corrector | Yu *et al.*  | ACM MM 2024   | [paper](https://dl.acm.org/doi/abs/10.1145/3664647.3688990?casa_token=SlLR5jgRRkgAAAAA:DzC124tFMWQSMYkKRGkPTwU-aaT7TSv_iVjE-dsZtbna9j3zCYX1A6qcfgmpEKTms8DoZDgplc5u8g) |


<!--
RAG-Guided Large Language Models for Visual Spatial Description with Adaptive Hallucination Corrector:  
Jun Yu, Yunxiang Zhang, Zerui Zhang, Zhao Yang, Gongpeng Zhao, Fengzhao Sun, Fanrui Zhang, Qingsong Liu, Jianqing Sun, Jiaen Liang, Yaohui Zhang
-->


#### 1.4 Multi-modal

| Title                                                                                                     | Authors                   | Venue/Date         | Paper Link                                                                                       |
|-----------------------------------------------------------------------------------------------------------|----------------------------|--------------------|--------------------------------------------------------------------------------------------------|
| VAT-KG: Knowledge-Intensive Multimodal Knowledge Graph Dataset for Retrieval-Augmented Generation | Park *et al.* | Arxiv 2025 (Jun) | [paper](https://arxiv.org/pdf/2506.21556) | 
| CoRe-MMRAG: Cross-Source Knowledge Reconciliation
for Multimodal RAG | Tian *et al.* | ACL 2025 (Jun) | [paper](https://aclanthology.org/2025.acl-long.1583.pdf) | 
| Evaluating VisualRAG: Quantifying Cross-Modal Performance in Enterprise Document Understanding | Mannam *et al.* | KDDW 2025 (Jun) | [paper](https://www.arxiv.org/pdf/2506.21604) | 
| Re-ranking Reasoning Context with Tree Search Makes Large Vision-Language Models Stronger | Yang *et al.* | Arxiv 2025 (Jun) | [paper](https://openreview.net/pdf?id=DJcEoC9JpQ) | 
| FlexRAG: A Flexible and Comprehensive Framework for Retrieval-Augmented Generation | Zhang *et al.* | Arxiv 2025 (Jun) | [paper](https://arxiv.org/pdf/2506.12494) | 
| MultiFinRAG: An Optimized Multimodal Retrieval-Augmented Generation (RAG) Framework for Financial Question Answering | Gondhalekar *et al.* | Arxiv 2025 (Jun) | [paper](https://arxiv.org/pdf/2506.20821) |
| DocReRank: Single-Page Hard Negative Query Generation for Training Multi-Modal RAG Rerankers | Wasserman *et al.* | Arxiv 2025 (May) | [paper](https://arxiv.org/pdf/2505.22584) | 
| A Multi-Granularity Retrieval Framework for Visually-Rich Documents | Xu *et al.* | Arxiv 2025 (May) | [paper](https://arxiv.org/pdf/2505.01457) | 
| Cross-modal RAG: Sub-dimensional Retrieval-Augmented Text-to-Image Generation | Zhu *et al.* | Arxiv 2025 (May) | [paper](https://arxiv.org/pdf/2505.21956) | 
| Benchmarking Multimodal Knowledge Conflict for Large Multimodal Models | Jia *et al.* | Arxiv 2025 (May) | [paper](https://arxiv.org/pdf/2505.19509) | 
| FinRAGBench-V: A Benchmark for Multimodal RAG with Visual Citation in the Financial Domain| Zhao *et al.* | Arxiv 2025 (May) | [paper](https://arxiv.org/pdf/2505.17471) | 
| RS-RAG: Bridging Remote Sensing Imagery and Comprehensive Knowledge with a Multi-Modal Dataset and Retrieval-Augmented Generation Model | Wen *et al.* | Arxiv 2025 (Apr) | [paper](https://arxiv.org/pdf/2504.04988) | 
| MMKB-RAG: A Multi-Modal Knowledge-Based Retrieval-Augmented Generation Framework | Ling *et al.* | Arxiv 2025 (Apr) | [paper](https://arxiv.org/pdf/2504.10074) | 
| HM-RAG: Hierarchical Multi-Agent Multimodal Retrieval Augmented Generation | Liu *et al.* | Arxiv 2025 (Apr) | [paper](https://arxiv.org/pdf/2504.12330) | 
| MRAMG-Bench: A Comprehensive Benchmark for Advancing Multimodal Retrieval-Augmented Multimodal Generation | Yu *et al.* | Arxiv 2025 (Apr) | [paper](https://arxiv.org/pdf/2502.04176) | 
| AutoStyle-TTS: Retrieval-Augmented Generation based Automatic Style Matching Text-to-Speech Synthesis | Luo *et al.* | ICME 2025 | [paper](https://arxiv.org/pdf/2504.10309) |
| RS-RAG: Bridging Remote Sensing Imagery and Comprehensive Knowledge with a Multi-Modal Dataset and Retrieval-Augmented Generation Model | Wen *et al.*           | Arxiv 2025 (Apr)   | [paper](https://arxiv.org/pdf/2503.13861) |
| RAD: Retrieval-Augmented Decision-Making of Meta-Actions with Vision-Language Models in Autonomous Driving | Wang *et al.*           | Arxiv 2025 (Mar)   | [paper](https://arxiv.org/pdf/2503.13861) |
| SuperRAG: Beyond RAG with Layout-Aware Graph Modeling | Yang *et al.*           | NACCL 2025 (Mar)   | [paper](https://arxiv.org/pdf/2503.04790) |
| MDocAgent: A Multi-Modal Multi-Agent Framework for Document Understanding | Han *et al.*           | Arxiv 2025 (Mar)   | [paper](https://arxiv.org/pdf/2503.13964) |
| SiQA: A Large Multi-Modal Question Answering Model for Structured Images Based on RAG | Liu *et al.*           | ICASSP 2025 (Mar)   | [paper](https://ieeexplore.ieee.org/abstract/document/10888359) |
| CommGPT: A Graph and Retrieval-Augmented Multimodal Communication Foundation Model  | Jiang *et al.*     | Arxiv 2025 (Feb)   | [paper](https://arxiv.org/pdf/2502.18763) |
| Benchmarking Multimodal RAG through a Chart-based Document Question-Answering Generation Framework | Yang *et al.*     | Arxiv 2025 (Feb)   | [paper](https://arxiv.org/pdf/2502.14864) |
| ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents  | Wang *et al.*     | Arxiv 2025 (Feb)   | [paper](https://arxiv.org/pdf/2502.18017) |
| Retrieval-Augmented Visual Question Answering via Built-in Autoregressive Search Engines | Long *et al.*     | Arxiv 2025 (Feb)   | [paper](https://arxiv.org/pdf/2502.16641) |
| WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models  | Chen *et al.*     | Arxiv 2025 (Feb)   | [paper](https://arxiv.org/pdf/2502.14727) |
| FilterRAG: Zero-Shot Informed Retrieval-Augmented Generation to Mitigate Hallucinations in VQA  | S M Sarwar  | Arxiv 2025 (Feb)   | [paper](https://arxiv.org/pdf/2502.18536) |
| A General Retrieval-Augmented Generation Framework for Multimodal Case-Based Reasoning Applications  | Ofir Marom           | Arxiv 2025 (Jan)   | [paper](https://arxiv.org/pdf/2501.05030) |
| Visual RAG: Expanding MLLM visual knowledge without fine-tuning  | Bonomo *et al.*        | Arxiv 2025 (Jan)   | [paper](https://arxiv.org/pdf/2501.10834) |
| Re-ranking the Context for Multimodal Retrieval Augmented Generation  | Mortaheb *et al.*        | Arxiv 2025 (Jan)   | [paper](https://arxiv.org/pdf/2501.04695) |
| MuKA: Multimodal Knowledge Augmented Visual Information-Seeking  | Deng *et al.*        | Coling 2025 (Jan)   | [paper](https://aclanthology.org/2025.coling-main.647.pdf) |
| mR2AG: Multimodal Retrieval-Reflection-Augmented Generation for Knowledge-Based VQA                      | Zhang *et al.*            | Arxiv 2024 (Nov)   | [paper](https://arxiv.org/pdf/2411.15041)                                                       |
| Wiki-LLaVA: Hierarchical Retrieval-Augmented Generation for Multimodal LLMs                               | Caffagni *et al.*         | CVPRW 2024         | [paper](https://openaccess.thecvf.com/content/CVPR2024W/MMFM/papers/Caffagni_Wiki-LLaVA_Hierarchical_Retrieval-Augmented_Generation_for_Multimodal_LLMs_CVPRW_2024_paper.pdf) |
| M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding              | Cho *et al.*              | Arxiv 2024 (Nov)   | [paper](https://arxiv.org/pdf/2410.21943)                                                       |
| UniRAG: Universal Retrieval Augmentation for Multi-Modal Large Language Models                            | Sharifymoghaddam *et al.*| Arxiv 2024 (Oct)   | [paper](https://arxiv.org/pdf/2405.10311)                                                       |
| MRAG-Bench: Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models                           | Hu *et al.*               | ICLR 2025 (Oct)    | [paper](https://arxiv.org/pdf/2410.08182)                                                       |
| VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents                           | Yu *et al.*               | ICLR 2025 (Oct)    | [paper](https://arxiv.org/pdf/2410.10594)                                                       |
| RoRA-VLM: Robust Retrieval Augmentation for Vision Language Models                                        | Qi *et al.*               | Arxiv 2024 (Oct)   | [paper](https://arxiv.org/pdf/2410.08876)                                                       |
| Beyond Text: Optimizing RAG with Multimodal Inputs for Industrial Applications                            | Riedler *et al.*          | Arxiv 2024 (Oct)   | [paper](https://arxiv.org/pdf/2410.21943)                                                       |
| SURf: Teaching Large Vision-Language Models to Selectively Utilize Retrieved Information                  | Sun *et al.*              | EMNLP 2024 (Sep)   | [paper](https://aclanthology.org/2024.emnlp-main.434.pdf)                                      |
| ColPali: Efficient Document Retrieval with Vision Language Models                                         | Faysse *et al.*           | ICLR 2025 (Jul)    | [paper](https://openreview.net/pdf?id=ogjBpZ8uSi)                                               |
| MLLM Is a Strong Reranker                                                                                 | Chen *et al.*             | Arxiv 2024 (Jul)   | [paper](https://arxiv.org/pdf/2409.14083)                                                       |
| RAVEN: Multitask Retrieval Augmented Vision-Language Learning                                             | Rao *et al.*              | COLM 2024 (Jun)    | [paper](https://openreview.net/pdf?id=GMalvQu0XL)                                               |
| SearchLVLMs                                                                                               | Li *et al.*               | NIPS 2024 (May)    | [paper](https://papers.nips.cc/paper_files/paper/2024/file/76954b4a44e158e738b4c64494977c6a-Paper-Conference.pdf) |
| UDKAG                                                                                                     | Li *et al.*               | CoRR 2024 (May)    | [paper](https://arxiv.org/abs/2405.14554v1)                                                     |
| Retrieval Meets Reasoning                                                                                 | Tan *et al.*              | Arxiv 2024 (Apr)   | [paper](https://arxiv.org/pdf/2405.20834)                                                       |
| RAR                                                                                                       | Liu *et al.*              | Arxiv 2024 (Mar)   |   [paper](https://arxiv.org/pdf/2403.13805)                                                        |
| MORE: Multi-mOdal REtrieval Augmented Generative Commonsense Reasoning  | Cui *et al.*         | ACL 2024 (Feb)   | [paper](https://aclanthology.org/2024.findings-acl.69.pdf) |
| Fine-grained Late-interaction Multi-modal Retrieval for RAG-VQA                                           | Lin *et al.*              | NIPS 2023 (Oct)    | [paper](https://papers.nips.cc/paper_files/paper/2023/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf) |
| Retrieval-based Knowledge Augmented Vision Language Pre-training                                          | Rao *et al.*              | ACMMM 2023 (Apr)   | [paper](https://arxiv.org/pdf/2304.13923)                                                       |
| ReVeaL                                                                                                    | Hu *et al.*               | CVPR 2023 (Apr)    | [paper](https://arxiv.org/pdf/2212.05221)                                                       |
| Murag                                                                                                     | Chen *et al.*             | EMNLP 2022 (Oct)   | [paper](https://aclanthology.org/2022.emnlp-main.375.pdf)                                      |

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
| Title                                                                                                       | Authors       | Venue/Date                                                                                  |
|-------------------------------------------------------------------------------------------------------------|---------------|---------------------------------------------------------------------------------------------|
| REALM: RAG-Driven Enhancement of Multimodal Electronic Health Records Analysis via Large Language Models                              | Zhu *et al.*  | Arxiv 2025 (Feb) [paper](https://arxiv.org/pdf/2402.07016)                                 |
| Mmed-rag: Versatile multimodal rag system for medical vision language models                               | Xia *et al.*  | Arxiv 2024 (Oct) [paper](https://arxiv.org/pdf/2410.13085)                                 |
| Rule: Reliable multimodal rag for factuality in medical vision language models                              | Xia *et al.*  | EMNLP 2024 [paper](https://aclanthology.org/2024.emnlp-main.62.pdf)                        |

<!--
Mmed-rag: Peng Xia, Kangyu Zhu, Haoran Li, Tianze Wang, Weijia Shi, Sheng Wang, Linjun Zhang, James Zou, Huaxiu Yao  
Rule: Peng Xia, Kangyu Zhu, Haoran Li, Hongtu Zhu, Yun Li, Gang Li, Linjun Zhang, Huaxiu Yao  
-->

### 2 Visual Generation 

#### 2.1 Image (Video) Generation

| Title                                                                                                   | Authors                   | Venue/Date       | Paper Link                                                                                       |
|---------------------------------------------------------------------------------------------------------|----------------------------|------------------|--------------------------------------------------------------------------------------------------|
| FairRAG: Fair Human Generation via Fair Retrieval Augmentation                                         | Shrestha *et al.*          | CVPR 2024        | [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Shrestha_FairRAG_Fair_Human_Generation_via_Fair_Retrieval_Augmentation_CVPR_2024_paper.pdf) |
| GarmentAligner (ECCV)                                                                                   | Zhang *et al.*             | ECCV 2025        | [paper](https://link.springer.com/chapter/10.1007/978-3-031-72698-9_9)                          |
| Retrieval-Augmented Diffusion Models                                                                    | Blattmann *et al.*         | NIPS 2022        | [paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/62868cc2fc1eb5cdf321d05b4b88510c-Paper-Conference.pdf) |
| Label-Retrieval-Augmented Diffusion Models                                                              | Chen *et al.*              | NIPS 2023        | [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/d191ba4c8923ed8fd8935b7c98658b5f-Paper-Conference.pdf) |
| CPR: Retrieval Augmented Generation for Copyright Protection                                            | Golatkar *et al.*          | CVPR 2023        | [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Golatkar_CPR_Retrieval_Augmented_Generation_for_Copyright_Protection_CVPR_2024_paper.pdf) |
| BrainRAM                                                                                                 | Xie *et al.*               | MM 2024          | [paper](https://dl.acm.org/doi/pdf/10.1145/3664647.3681296)                                    |
| Animate-A-Story                                                                                         | He *et al.*                | Arxiv 2023       | [paper](https://arxiv.org/pdf/2307.06940)                                                       |
| RealGen                                                                                                  | Ding *et al.*              | ECCV 2024        | [paper](https://arxiv.org/pdf/2312.13303)                                                       |
| Grounding Language Models for Visual Entity Recognition                                                 | Xiao *et al.*              | ECCV 2024        | [paper](https://arxiv.org/pdf/2402.18695)                                                       |
| GarmentAligner (Arxiv)                                                                                  | Zhang *et al.*             | ECCV 2024        | [paper](https://arxiv.org/pdf/2408.12352)                                                       |
| Retrieval-Augmented Layout Transformer                                                                  | Horita *et al.*            | CVPR 2024        | [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Horita_Retrieval-Augmented_Layout_Transformer_for_Content-Aware_Layout_Generation_CVPR_2024_paper.pdf) |
| The Neglected Tails in Vision-Language Models                                                           | Parashar *et al.*          | CVPR 2024        | [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Horita_Retrieval-Augmented_Layout_Transformer_for_Content-Aware_Layout_Generation_CVPR_2024_paper.pdf) |
| Prompt Expansion for Adaptive Text-to-Image Generation                                                  | Datta *et al.*             | ACL 2024         | [paper](https://arxiv.org/pdf/2312.16720)                                                       |
| Factuality Tax of Diversity-Intervened Generation                                                       | Wan *et al.*               | EMNLP 2024       | [paper](https://arxiv.org/pdf/2407.00377)                                                       |
| Diffusion-Based Augmentation for Captioning and Retrieval                                               | Cioni *et al.*             | ICCV 2023        | [paper](https://openaccess.thecvf.com/content/ICCV2023W/e-Heritage/papers/Cioni_Diffusion_Based_Augmentation_for_Captioning_and_Retrieval_in_Cultural_Heritage_ICCVW_2023_paper.pdf) |
| ReMoDiffuse                                                                                              | Zhang *et al.*             | ICCV 2023        | [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_ReMoDiffuse_Retrieval-Augmented_Motion_Diffusion_Model_ICCV_2023_paper.pdf) |
| Re-imagen                                                                                                | Chen *et al.*              | Arxiv 2022       | [paper](https://arxiv.org/pdf/2209.14491)                                                       |
| Instruct-Imagen                                                                                         | Hu *et al.*                | CVPR 2024        | [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Hu_Instruct-Imagen_Image_Generation_with_Multi-modal_Instruction_CVPR_2024_paper.pdf) |
| ImageRAG                                                                                                 | Shalev-Arkushin *et al.*   | Arxiv 2025       | [paper](https://arxiv.org/pdf/2502.09411)                                                       |
| FineRAG                                                                                                  | Yuan *et al.*              | COLING 2025      | [paper](https://aclanthology.org/2025.coling-main.741.pdf)                                     |
| RealRAG                                                                                                  | Lyu *et al.*               | Arxiv 2025       | [paper](https://arxiv.org/pdf/2502.00848)                                                       |


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
| Title                                                                                                   | Authors         | Venue/Date        | Paper Link                                                                                       |
|---------------------------------------------------------------------------------------------------------|------------------|-------------------|--------------------------------------------------------------------------------------------------|
| Phidias: A Generative Model for Creating 3D Content from Text, Image, and 3D Conditions                | Wang *et al.*    | Arxiv 2024 (Sep)  | [paper](https://arxiv.org/pdf/2409.11406)                                                       |
| Retrieval-Augmented Score Distillation for Text-to-3D Generation                                       | Seo *et al.*     | ICML 2024         | [paper](https://arxiv.org/pdf/2402.02972)                                                       |
| Diorama: Unleashing Zero-shot Single-view 3D Scene Modeling                                            | Wu *et al.*      | Arxiv 2024 (Nov)  | [paper](https://arxiv.org/pdf/2411.19492)                                                       |
| Interaction-based Retrieval-augmented Diffusion for Protein 3D Generation                              | Huang *et al.*   | ICML 2024         | [paper](https://openreview.net/pdf?id=eejhD9FCP3)                                               |
| ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model                                                | Zhang *et al.*   | ICCV 2023         | [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_ReMoDiffuse_Retrieval-Augmented_Motion_Diffusion_Model_ICCV_2023_paper.pdf) |

<!--
Phidias: Zhenwei Wang, Tengfei Wang, Zexin He, Gerhard Hancke, Ziwei Liu, Rynson W.H. Lau  
Retrieval-Augmented Score Distillation: Junyoung Seo, Susung Hong, Wooseok Jang, Ines Hyeonsu Kim, Minseop Kwak, Doyup Lee, Seungryong Kim  
Diorama: Qirui Wu, Denys Iliash, Daniel Ritchie, Manolis Savva, Angel X. Chang  
Interaction-based Diffusion: Zhilin Huang, Ling Yang, Xiangxin Zhou, Chujun Qin, Yijie Yu, Xiawu Zheng, Zikun Zhou, Wentao Zhang, Yu Wang, Wenming Yang  
ReMoDiffuse: Mingyuan Zhang, Xinying Guo, Liang Pan, Zhongang Cai, Fangzhou Hong, Huirong Li, Lei Yang, Ziwei Liu  
-->


### 3. Embodied AI
| Title                                                                                         | Authors        | Venue/Date        | Paper Link                                                                                      |
|------------------------------------------------------------------------------------------------|----------------|-------------------|--------------------------------------------------------------------------------------------------|
| RAG-6DPose: Retrieval-Augmented 6D Pose Estimation via Leveraging CAD as Knowledge Base | Wang *et al.* | IROS 2025 July | [paper](https://arxiv.org/pdf/2506.18856) | 
| RAD: Retrieval-Augmented Decision-Making of Meta-Actions with Vision-Language Models in Autonomous Driving | Wang *et al.* | Arxiv 2025 (Mar) | [paper](https://arxiv.org/pdf/2503.13861)                                                |
| RANa: Retrieval-Augmented Navigation      | Monaci *et al.*     | Arxiv 2025 (Apr)       | [paper]([https://dl.acm.org/doi/pdf/10.1145/3664647.3680661](https://arxiv.org/pdf/2504.03524))                                     |
| P-RAG: Progressive Retrieval Augmented Generation For Planning on Embodied Everyday Task      | Xu *et al.*     | ACM MM 2024       | [paper](https://dl.acm.org/doi/pdf/10.1145/3664647.3680661)                                     |
| Realgen: Retrieval Augmented Generation for Controllable Traffic Scenarios                   | Ding *et al.*   | ECCV 2024         | [paper](https://arxiv.org/pdf/2312.13303)                                                       |
| Retrieval-Augmented Embodied Agents                                                           | Zhu *et al.*    | CVPR 2024         | [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_Retrieval-Augmented_Embodied_Agents_CVPR_2024_paper.pdf) |
| ENWAR: A RAG-empowered Multi-Modal LLM Framework for Wireless Environment Perception          | Nazar *et al.*  | Arxiv 2024 (Oct)  | [paper](https://arxiv.org/pdf/2410.18104)                                                       |
| Embodied-RAG: General Non-parametric Embodied Memory for Retrieval and Generation             | Xie *et al.*    | Arxiv 2024 (Oct)  | [paper](https://arxiv.org/pdf/2409.18313)                                                       |
| RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning   | Yuan *et al.*   | Arxiv 2024 (May)  | [paper](https://arxiv.org/abs/2402.10828)                                                       |

<!--
P-RAG: Weiye Xu, Min Wang, Wengang Zhou, Houqiang Li  
Realgen: Wenhao Ding, Yulong Cao, Ding Zhao, Chaowei Xiao, Marco Pavone  
Retrieval-Augmented Embodied Agents: Yichen Zhu, Zhicai Ou, Xiaofeng Mou, Jian Tang  
ENWAR: Ahmad M. Nazar, Abdulkadir Celik, Mohamed Y. Selim, Asmaa Abdallah, Daji Qiao, Ahmed M. Eltawil  
Embodied-RAG: Quanting Xie, So Yeon Min, Tianyi Zhang, Kedi Xu, Aarav Bajaj, Ruslan Salakhutdinov, Matthew Johnson-Roberson, Yonatan Bisk  
RAG-Driver: Jianhao Yuan, Shuyang Sun, Daniel Omeiza, Bo Zhao, Paul Newman, Lars Kunze, Matthew Gadd  
-->

