# 全部参考文献 DBLP 复核 — 2026-11-16

核验范围：当前 ACL 稿 `acl_submission/main.tex`（含附录）实际引用的全部 **84 篇**，及对应 `references.bib`。本报告不把 GitHub 的 177 篇目录或旧 IEEE 完整稿混入统计。

## 结果

- **84/84** 条 DBLP 记录由官方实时 SPARQL 查询返回；没有只凭手写 DBLP 链接或旧审计文件认定存在。
- **84/84** 题名匹配，**84/84** 完整作者顺序匹配；比较仅忽略标点、大小写、TeX 重音编码及 DBLP 作者消歧编号。
- **82/84** 年份与对应 DBLP 记录一致。另两篇为“DBLP 预印本身份 + 官方正式发表版本”，详见下文。
- 文内引用键、外部 BibTeX、编辑器内嵌参考文献均为同一组 **84** 条；无缺失、未引用或重复的文献身份。
- 所有已填写 DOI 均与 DBLP 一致；有可比页码的条目也一致。会议分卷／年份和 arXiv／CoRR 的显示差异已区分。

据此，**当前 84 条引用未发现虚构论文、错配题名或错配作者**。本次核验的是文献身份与书目信息，不能单独证明稿件对每项实验结果的解读正确。

## 两项版本差异

| 文献 | DBLP 返回的记录 | 稿件采用的正式版本 |
|---|---|---|
| MiRAGE | [2025 预印本](https://dblp.org/rec/journals/corr/abs-2510-24870) | [EMNLP 2026](https://aclanthology.org/2026.emnlp-main.1765/) |
| UniDoc-Bench | [2025 预印本](https://dblp.org/rec/journals/corr/abs-2510-03663) | [Findings of EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.534/) |

两篇的正式题名、完整作者顺序、年份和页码均另与 ACL Anthology 导出引用核对。DBLP 检查没有建立对应的正式会议记录，故不声称这两条的会议版元数据来自 DBLP。MiRAGE 的 HTML `citation_author` 元数据省略 William Gantt Walden 的中间名；页面可见引用及导出 BibTeX 包含该中间名，与 DBLP／稿件一致，保留完整署名。

没有发现需要改正的正文或参考文献身份，故保留已经核实的正式出版引用。

## 复核证据

- `audit.json`：逐条题名、完整作者、年份、会议信息及差异判定。
- `references-check.csv`：便于筛选的逐条检查表。
- `bibliography-input.json`：本次被核验的 BibTeX 字段快照；原文件 SHA256 记录在 `audit.json`。
- `properties-1..4.rq/json`、`authors-1..4.rq/json`：本次实际执行的八个查询与原始返回。
- `formal-publication-checks.json`：两个官方出版页面的引用元数据及页面哈希；不转载页面正文。
- `manifest.json`：证据文件 SHA256 与大小。

服务为 [DBLP 官方 SPARQL](https://sparql.dblp.org/)，未把 Scholar 搜索页或一个看似合法的 URL 当作核验结果。DBLP 常规网页在工具中不可直接访问；本次通过其官方结构化知识图谱获取全部记录。

## 逐条清单

| 引用键 | 题名 | DBLP | 核验 |
|---|---|---|---|
| `arxiv250208826` | Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation | [记录](https://dblp.org/rec/conf/acl/AbootorabiZDMMG25) | 题名／作者／年份匹配 |
| `arxiv250408748` | A Survey of Multimodal Retrieval-Augmented Generation | [记录](https://dblp.org/rec/journals/corr/abs-2504-08748) | 题名／作者／年份匹配 |
| `arxiv250524073` | mRAG: Elucidating the Design Space of Multi-modal Retrieval-Augmented Generation | [记录](https://dblp.org/rec/journals/corr/abs-2505-24073) | 题名／作者／年份匹配 |
| `arxiv260304238` | Retrieval or Representation? Reassessing Benchmark Gaps in Multilingual and Visually Rich RAG | [记录](https://dblp.org/rec/journals/corr/abs-2603-04238) | 题名／作者／年份匹配 |
| `arxiv260500911` | When Good OCR Is Not Enough: Benchmarking OCR Robustness for Retrieval-Augmented Generation | [记录](https://dblp.org/rec/conf/acl/SunWHZJCZ26) | 题名／作者／年份匹配 |
| `arxiv260529956` | Uncertainty Quantification for Multimodal Retrieval Augmented Generation | [记录](https://dblp.org/rec/conf/ictir/BinzSH26) | 题名／作者／年份匹配 |
| `ataallah2024goldfish` | Goldfish: Vision-Language Understanding of Arbitrarily Long Videos | [记录](https://dblp.org/rec/conf/eccv/AtaallahSASZDZSE24) | 题名／作者／年份匹配 |
| `chen2022murag` | MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text | [记录](https://dblp.org/rec/conf/emnlp/ChenHCVC22) | 题名／作者／年份匹配 |
| `cho2024m3docrag` | M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding | [记录](https://dblp.org/rec/journals/corr/abs-2411-04952) | 题名／作者／年份匹配 |
| `ding2024realgen` | RealGen: Retrieval Augmented Generation for Controllable Traffic Scenarios | [记录](https://dblp.org/rec/conf/eccv/DingCZXP24) | 题名／作者／年份匹配 |
| `faysse2024colpali` | ColPali: Efficient Document Retrieval with Vision Language Models | [记录](https://dblp.org/rec/conf/iclr/FaysseSWOVHC25) | 题名／作者／年份匹配 |
| `fu2024framefusion` | FrameFusion: Combining Similarity and Importance for Video Token Reduction on Large Vision Language Models | [记录](https://dblp.org/rec/conf/iccv/FuLHDYYNW25) | 题名／作者／年份匹配 |
| `he2023animate` | Animate-A-Story: Storytelling with Retrieval-Augmented Video Generation | [记录](https://dblp.org/rec/journals/corr/abs-2307-06940) | 题名／作者／年份匹配 |
| `jeong2025videorag` | VideoRAG: Retrieval-Augmented Generation over Video Corpus | [记录](https://dblp.org/rec/conf/acl/JeongKBH25) | 题名／作者／年份匹配 |
| `kim2026taxonomy` | A Taxonomy of Knowledge Bases for Retrieval-Augmented Methods in Vision: A Comprehensive Survey | [记录](https://dblp.org/rec/journals/access/KimLY26) | 题名／作者／年份匹配 |
| `lewis2020retrieval` | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | [记录](https://dblp.org/rec/conf/nips/LewisPPPKGKLYR020) | 题名／作者／年份匹配 |
| `lin2023finegrained` | Fine-grained Late-interaction Multi-modal Retrieval for Retrieval Augmented Visual Question Answering | [记录](https://dblp.org/rec/conf/nips/LinCMCB23) | 题名／作者／年份匹配 |
| `mathew2021docvqa` | DocVQA: A Dataset for VQA on Document Images | [记录](https://dblp.org/rec/conf/wacv/MathewKJ21) | 题名／作者／年份匹配 |
| `shalev2025imagerag` | ImageRAG: Dynamic Image Retrieval for Reference-Guided Image Generation | [记录](https://dblp.org/rec/conf/iclr/Shalev-Arkushin26) | 题名／作者／年份匹配 |
| `wang2024phidias` | Phidias: A Generative Model for Creating 3D Content from Text, Image, and 3D Conditions with Reference-Augmented Diffusion | [记录](https://dblp.org/rec/conf/iclr/00030HH0L25) | 题名／作者／年份匹配 |
| `wu2025visualrag` | Visual-RAG: Benchmarking Text-to-Image Retrieval Augmented Generation for Visual Knowledge Intensive Queries | [记录](https://dblp.org/rec/conf/sigir/WuLLYW26) | 题名／作者／年份匹配 |
| `xie2024embodied` | Embodied-RAG: General non-parametric Embodied Memory for Retrieval and Generation | [记录](https://dblp.org/rec/journals/corr/abs-2409-18313) | 题名／作者／年份匹配 |
| `xu2024p` | P-RAG: Progressive Retrieval Augmented Generation For Planning on Embodied Everyday Task | [记录](https://dblp.org/rec/conf/mm/XuWZL24) | 题名／作者／年份匹配 |
| `yi2024video` | Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models | [记录](https://dblp.org/rec/conf/cvpr/YiWLNG25) | 题名／作者／年份匹配 |
| `yu2024visrag` | VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents | [记录](https://dblp.org/rec/conf/iclr/YuTXCRYLWHL025) | 题名／作者／年份匹配 |
| `mahata2026midr` | MIDR: Enrichment-Augmented Indexing for Multimodal Document Retrieval | [记录](https://dblp.org/rec/journals/corr/abs-2609-01316) | 题名／作者／年份匹配 |
| `wang2026trajrag` | TrajRAG: Retrieving Geometric-Semantic Experience for Zero-Shot Object Navigation | [记录](https://dblp.org/rec/conf/cvpr/WangZZSDJ26) | 题名／作者／年份匹配 |
| `hu2026longliverag` | LongLive-RAG: A General Retrieval-Augmented Framework for Long Video Generation | [记录](https://dblp.org/rec/journals/corr/abs-2606-02553) | 题名／作者／年份匹配 |
| `lee2026carve` | Rethinking RAG in Long Videos: What to Retrieve and How to Use It? | [记录](https://dblp.org/rec/journals/corr/abs-2606-13141) | 题名／作者／年份匹配 |
| `wang2026graphdoc` | Multimodal Graph RAG for Long-range Visually Rich Document Understanding | [记录](https://dblp.org/rec/journals/corr/abs-2606-28780) | 题名／作者／年份匹配 |
| `chakraborty2026marquis` | MARQUIS: A Three-Stage Pipeline for Video Retrieval-Augmented Generation | [记录](https://dblp.org/rec/journals/corr/abs-2605-17640) | 题名／作者／年份匹配 |
| `chen2026berag` | BERAG: Bayesian Ensemble Retrieval-Augmented Generation for Knowledge-based Visual Question Answering | [记录](https://dblp.org/rec/journals/corr/abs-2604-22678) | 题名／作者／年份匹配 |
| `gao2026scaling` | Scaling Beyond Context: A Survey of Multimodal Retrieval-Augmented Generation for Document Understanding | [记录](https://dblp.org/rec/conf/acl/GaoZJDCCLZBG26) | 题名／作者／年份匹配 |
| `gao2023retrieval` | Retrieval-Augmented Generation for Large Language Models: A Survey | [记录](https://dblp.org/rec/journals/corr/abs-2312-10997) | 题名／作者／年份匹配 |
| `fan2024surveyragmeetingllms` | A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models | [记录](https://dblp.org/rec/conf/kdd/FanDNWLYCL24) | 题名／作者／年份匹配 |
| `liu2024rar` | RAR: Retrieving and Ranking Augmented MLLMs for Visual Recognition | [记录](https://dblp.org/rec/journals/tip/LiuSZLZDXLW26) | 题名／作者／年份匹配 |
| `long2022retrieval` | Retrieval Augmented Classification for Long-Tail Visual Recognition | [记录](https://dblp.org/rec/conf/cvpr/Long0ANPGBSH22) | 题名／作者／年份匹配 |
| `kim2024retrieval` | Retrieval-Augmented Open-Vocabulary Object Detection | [记录](https://dblp.org/rec/conf/cvpr/KimCKK24) | 题名／作者／年份匹配 |
| `li2024evcap` | Evcap: Retrieval-Augmented Image Captioning with External Visual-Name Memory for Open-World Comprehension | [记录](https://dblp.org/rec/conf/cvpr/LiVSN24) | 题名／作者／年份匹配 |
| `ramos2023retrieval` | Retrieval-augmented Image Captioning | [记录](https://dblp.org/rec/conf/eacl/RamosEM23) | 题名／作者／年份匹配 |
| `li2024understanding` | Understanding Retrieval Robustness for Retrieval-augmented Image Captioning | [记录](https://dblp.org/rec/conf/acl/LiLRTE24) | 题名／作者／年份匹配 |
| `hu2023reveal` | Reveal: Retrieval-Augmented Visual-Language Pre-Training with Multi-Source Multimodal Knowledge Memory | [记录](https://dblp.org/rec/conf/cvpr/HuI0WCSSRF23) | 题名／作者／年份匹配 |
| `rao2023retrieval` | Retrieval-based Knowledge Augmented Vision Language Pre-training | [记录](https://dblp.org/rec/conf/mm/RaoSLZY23) | 题名／作者／年份匹配 |
| `liu2023learning` | Learning Customized Visual Models with Retrieval-Augmented Knowledge | [记录](https://dblp.org/rec/conf/cvpr/LiuSY0GLL23) | 题名／作者／年份匹配 |
| `li2024search` | SearchLVLMs: A Plug-and-Play Framework for Augmenting Large Vision-Language Models by Searching Up-to-Date Internet Knowledge | [记录](https://dblp.org/rec/conf/nips/Li0JLS000Z24) | 题名／作者／年份匹配 |
| `sun2024surf` | SURf: Teaching Large Vision-Language Models to Selectively Utilize Retrieved Information | [记录](https://dblp.org/rec/conf/emnlp/SunZZSQ024) | 题名／作者／年份匹配 |
| `qi2024roravlm` | RoRA-VLM: Robust Retrieval-Augmented Vision Language Models | [记录](https://dblp.org/rec/journals/corr/abs-2410-08876) | 题名／作者／年份匹配 |
| `rao2024raven` | RAVEN: Multitask Retrieval Augmented Vision-Language Learning | [记录](https://dblp.org/rec/journals/corr/abs-2406-19150) | 题名／作者／年份匹配 |
| `chen2024mllm` | MLLM Is a Strong Reranker: Advancing Multimodal Retrieval-augmented Generation via Knowledge-enhanced Reranking and Noise-injected Training | [记录](https://dblp.org/rec/journals/corr/abs-2407-21439) | 题名／作者／年份匹配 |
| `tan2024retrieval` | Retrieval Meets Reasoning: Even High-school Textbook Knowledge Benefits Multimodal Reasoning | [记录](https://dblp.org/rec/journals/corr/abs-2405-20834) | 题名／作者／年份匹配 |
| `xia2024rule` | RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models | [记录](https://dblp.org/rec/conf/emnlp/XiaZLZLLZY24) | 题名／作者／年份匹配 |
| `xia2024mmed` | MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models | [记录](https://dblp.org/rec/conf/iclr/0005ZLWSWZ0Y25) | 题名／作者／年份匹配 |
| `luo2024video` | Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension | [记录](https://dblp.org/rec/conf/nips/LuoZLYLFHJCLJ25) | 题名／作者／年份匹配 |
| `arefeen2024irag` | iRAG: Advancing RAG for Videos with an Incremental Approach | [记录](https://dblp.org/rec/conf/cikm/ArefeenDUC24) | 题名／作者／年份匹配 |
| `Arefeen2024ViTA` | ViTA: An Efficient Video-to-Text Algorithm using VLM for RAG-based Video Analysis System | [记录](https://dblp.org/rec/conf/cvpr/ArefeenDUC22) | 题名／作者／年份匹配 |
| `xiong2025streaming` | Streaming Video Understanding and Multi-round Interaction with Memory-enhanced Knowledge | [记录](https://dblp.org/rec/conf/iclr/XiongYYZ0ZL25) | 题名／作者／年份匹配 |
| `zhang2024omagent` | OmAgent: A Multi-modal Agent Framework for Complex Video Understanding with Task Divide-and-Conquer | [记录](https://dblp.org/rec/conf/emnlp/ZhangZYML24) | 题名／作者／年份匹配 |
| `sankaradas2025streamingragrealtimecontextualretrieval` | StreamingRAG: Real-time Contextual Retrieval and Generation Framework | [记录](https://dblp.org/rec/journals/corr/abs-2501-14101) | 题名／作者／年份匹配 |
| `lu2022learnexplainmultimodal` | Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering | [记录](https://dblp.org/rec/conf/nips/LuMX0CZTCK22) | 题名／作者／年份匹配 |
| `marino2019okvqa` | OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge | [记录](https://dblp.org/rec/conf/cvpr/MarinoRFM19) | 题名／作者／年份匹配 |
| `10.1007/978-3-031-20074-8_9` | A-OKVQA: A Benchmark for Visual Question Answering Using World Knowledge | [记录](https://dblp.org/rec/conf/eccv/SchwenkKCMM22) | 题名／作者／年份匹配 |
| `chang2022webqa` | WebQA: Multihop and Multimodal QA | [记录](https://dblp.org/rec/conf/cvpr/ChangCNGSB22) | 题名／作者／年份匹配 |
| `chen2023pretrainedvision` | Can Pre-trained Vision and Language Models Answer Visual Information-Seeking Questions? | [记录](https://dblp.org/rec/conf/emnlp/ChenHLSCRC23) | 题名／作者／年份匹配 |
| `hu2023opendomain` | Open-domain Visual Entity Recognition: Towards Recognizing Millions of Wikipedia Entities | [记录](https://dblp.org/rec/conf/iccv/HuLCKJLTC23) | 题名／作者／年份匹配 |
| `dong2025mmdocir` | MMDocIR: Benchmarking Multimodal Retrieval for Long Documents | [记录](https://dblp.org/rec/conf/emnlp/DongCDLTL25) | 题名／作者／年份匹配 |
| `hu2025mragbench` | MRAG-Bench: Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models | [记录](https://dblp.org/rec/conf/iclr/0006GDFLC025) | 题名／作者／年份匹配 |
| `wasserman2025realmmra` | REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark | [记录](https://dblp.org/rec/conf/acl/WassermanPNGSBK25) | 题名／作者／年份匹配 |
| `mortaheb2025rag` | RAG-Check: Evaluating Multimodal Retrieval Augmented Generation Performance | [记录](https://dblp.org/rec/journals/corr/abs-2501-03995) | 题名／作者／年份匹配 |
| `lyu2025realrag` | RealRAG: Retrieval-augmented Realistic Image Generation via Self-reflective Contrastive Learning | [记录](https://dblp.org/rec/conf/icml/Lyu0JY000H25) | 题名／作者／年份匹配 |
| `horita2024retrieval` | Retrieval-Augmented Layout Transformer for Content-Aware Layout Generation | [记录](https://dblp.org/rec/conf/cvpr/HoritaIKYA24) | 题名／作者／年份匹配 |
| `zhang2023remodiffuse` | ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model | [记录](https://dblp.org/rec/conf/iccv/ZhangGPCHLYL23) | 题名／作者／年份匹配 |
| `yuan2025finerag` | FineRAG: Fine-grained Retrieval-Augmented Text-to-Image Generation | [记录](https://dblp.org/rec/conf/coling/YuanZWXNLD25) | 题名／作者／年份匹配 |
| `zhang2025garmentaligner` | GarmentAligner: Text-to-Garment Generation via Retrieval-Augmented Multi-level Corrections | [记录](https://dblp.org/rec/conf/eccv/ZhangCZLCYL24) | 题名／作者／年份匹配 |
| `wan2024factuality` | The Factuality Tax of Diversity-Intervened Text-to-Image Generation: Benchmark and Fact-Augmented Intervention | [记录](https://dblp.org/rec/conf/emnlp/WanWWC24) | 题名／作者／年份匹配 |
| `golatkar2024cpr` | CPR: Retrieval Augmented Generation for Copyright Protection | [记录](https://dblp.org/rec/conf/cvpr/GolatkarAZ0SS24) | 题名／作者／年份匹配 |
| `seo2024retrieval` | Retrieval-Augmented Score Distillation for Text-to-3D Generation | [记录](https://dblp.org/rec/conf/icml/SeoHJKKLK24) | 题名／作者／年份匹配 |
| `wu2024dioramaunleashingzeroshotsingleview` | Diorama: Unleashing Zero-Shot Single-View 3D Indoor Scene Modeling | [记录](https://dblp.org/rec/conf/iccv/WuIRSC25) | 题名／作者／年份匹配 |
| `yuan2024rag` | RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning in Multi-Modal Large Language Model | [记录](https://dblp.org/rec/journals/corr/abs-2402-10828) | 题名／作者／年份匹配 |
| `nazar2024enwar` | ENWAR: A RAG-Empowered Multi-Modal LLM Framework for Wireless Environment Perception | [记录](https://dblp.org/rec/journals/cm/NazarCSAQE26) | 题名／作者／年份匹配 |
| `blattmann2022retrieval` | Retrieval-Augmented Diffusion Models | [记录](https://dblp.org/rec/conf/nips/BlattmannROMO22) | 题名／作者／年份匹配 |
| `sun2025visrag2` | VisRAG 2.0: Evidence-Guided Multi-Image Reasoning in Visual Retrieval-Augmented Generation | [记录](https://dblp.org/rec/journals/corr/abs-2510-09733) | 题名／作者／年份匹配 |
| `martin2026mirage` | Seeing Through the MiRAGE: Evaluating Multimodal Retrieval Augmented Generation | [记录](https://dblp.org/rec/journals/corr/abs-2510-24870) | 身份匹配；正式版本另核验 |
| `hu2026docrefrag` | Doc-REFRAG: Rethinking Multimodal Document Retrieval-Augmented Generation | [记录](https://dblp.org/rec/conf/emnlp/HuXHYZLJZ26) | 题名／作者／年份匹配 |
| `peng2026unidoc` | UNIDOC-BENCH: A Unified Benchmark for Document-Centric Multimodal RAG | [记录](https://dblp.org/rec/journals/corr/abs-2510-03663) | 身份匹配；正式版本另核验 |
