# Bibliographic verification, 15 October 2026

All 81 references retained in the ACL manuscript have a retrieved record from
dblp's official public knowledge graph. The raw JSON responses, queries, and
ordered author signatures are saved in `metadata/dblp-audit-20261015/`. Each BibTeX entry carries
its `biburl`. Conference and preprint versions count as one paper; an indexed
proceedings or journal version is preferred. Conference event year is used when
available; the raw publication year is preserved separately (ECCV volumes can
have a later imprint year). No venue is inferred from an arXiv acceptance note.

Service: [dblp public SPARQL API documentation](https://blog.dblp.org/2024/09/09/introducing-our-public-sparql-query-service/).
Queries were executed against `https://sparql.dblp.org/sparql` on 2026-10-15.
The regular dblp website/API returned a bot challenge; the documented public
SPARQL endpoint returned structured records. No challenge was bypassed.

This verifies bibliographic identity and metadata, not experimental validity or
every scientific assertion. The review remains a targeted review, with incomplete
full-text coding and no new experiments. The bibliography uses 21 preprints and
60 proceedings/journal records (derive current counts from verified-records.json
if this set changes). Google Scholar search URLs were not treated as verification.

## Corrections and unresolved item

- The diffusion citation was incorrectly titled “Semi-Parametric Neural Image
  Synthesis” in the preceding ACL draft. It is corrected to **Retrieval-Augmented
  Diffusion Models**, with all five authors and the NeurIPS 2022 record.
- Formal records replace stale preprint metadata where available, including
  Ask in Any Modality, ImageRAG, Visual-RAG, Video-RAG, MMed-RAG, RAR and ENWAR.
- Complementary Retrieval-Augmented Prompting for Consistent Long-Form Video
  Generation (arXiv:2609.37407) was present in the preceding draft. An arXiv record
  was found earlier, but no matching dblp record or direct Google Scholar record
  was established in this pass. It is excluded from the formal manuscript pending
  the user's requested database verification. This is not a finding that the
  paper does not exist.
- Exact-title matches were preferred. Diorama's proceedings title contains
  “Indoor”; the version change is explicit in the retrieved record.
- A short-title difference in MMDocIR was checked against the matching preprint
  and proceedings records. Citation keys are stable internal identifiers, not
  assertions about publication year.

## Retained papers

| Citation key | Title | Year / status | DBLP |
|---|---|---|---|
| `arxiv250408748` | A Survey of Multimodal Retrieval-Augmented Generation | 2025 / preprint | [record](https://dblp.org/rec/journals/corr/abs-2504-08748) |
| `fan2024surveyragmeetingllms` | A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models | 2024 / proceedings | [record](https://dblp.org/rec/conf/kdd/FanDNWLYCL24) |
| `kim2026taxonomy` | A Taxonomy of Knowledge Bases for Retrieval-Augmented Methods in Vision: A Comprehensive Survey | 2026 / journal | [record](https://dblp.org/rec/journals/access/KimLY26) |
| `10.1007/978-3-031-20074-8_9` | A-OKVQA: A Benchmark for Visual Question Answering Using World Knowledge | 2022 / proceedings | [record](https://dblp.org/rec/conf/eccv/SchwenkKCMM22) |
| `he2023animate` | Animate-A-Story: Storytelling with Retrieval-Augmented Video Generation | 2023 / preprint | [record](https://dblp.org/rec/journals/corr/abs-2307-06940) |
| `arxiv250208826` | Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation | 2025 / proceedings | [record](https://dblp.org/rec/conf/acl/AbootorabiZDMMG25) |
| `chen2026berag` | BERAG: Bayesian Ensemble Retrieval-Augmented Generation for Knowledge-based Visual Question Answering | 2026 / preprint | [record](https://dblp.org/rec/journals/corr/abs-2604-22678) |
| `chen2023pretrainedvision` | Can Pre-trained Vision and Language Models Answer Visual Information-Seeking Questions? | 2023 / proceedings | [record](https://dblp.org/rec/conf/emnlp/ChenHLSCRC23) |
| `faysse2024colpali` | ColPali: Efficient Document Retrieval with Vision Language Models | 2025 / proceedings | [record](https://dblp.org/rec/conf/iclr/FaysseSWOVHC25) |
| `golatkar2024cpr` | CPR: Retrieval Augmented Generation for Copyright Protection | 2024 / proceedings | [record](https://dblp.org/rec/conf/cvpr/GolatkarAZ0SS24) |
| `wu2024dioramaunleashingzeroshotsingleview` | Diorama: Unleashing Zero-Shot Single-View 3D Indoor Scene Modeling | 2025 / proceedings | [record](https://dblp.org/rec/conf/iccv/WuIRSC25) |
| `mathew2021docvqa` | DocVQA: A Dataset for VQA on Document Images | 2021 / proceedings | [record](https://dblp.org/rec/conf/wacv/MathewKJ21) |
| `xie2024embodied` | Embodied-RAG: General non-parametric Embodied Memory for Retrieval and Generation | 2024 / preprint | [record](https://dblp.org/rec/journals/corr/abs-2409-18313) |
| `nazar2024enwar` | ENWAR: A RAG-Empowered Multi-Modal LLM Framework for Wireless Environment Perception | 2026 / journal | [record](https://dblp.org/rec/journals/cm/NazarCSAQE26) |
| `li2024evcap` | Evcap: Retrieval-Augmented Image Captioning with External Visual-Name Memory for Open-World Comprehension | 2024 / proceedings | [record](https://dblp.org/rec/conf/cvpr/LiVSN24) |
| `lin2023finegrained` | Fine-grained Late-interaction Multi-modal Retrieval for Retrieval Augmented Visual Question Answering | 2023 / proceedings | [record](https://dblp.org/rec/conf/nips/LinCMCB23) |
| `yuan2025finerag` | FineRAG: Fine-grained Retrieval-Augmented Text-to-Image Generation | 2025 / proceedings | [record](https://dblp.org/rec/conf/coling/YuanZWXNLD25) |
| `fu2024framefusion` | FrameFusion: Combining Similarity and Importance for Video Token Reduction on Large Vision Language Models | 2025 / proceedings | [record](https://dblp.org/rec/conf/iccv/FuLHDYYNW25) |
| `zhang2025garmentaligner` | GarmentAligner: Text-to-Garment Generation via Retrieval-Augmented Multi-level Corrections | 2024 / proceedings | [record](https://dblp.org/rec/conf/eccv/ZhangCZLCYL24) |
| `ataallah2024goldfish` | Goldfish: Vision-Language Understanding of Arbitrarily Long Videos | 2024 / proceedings | [record](https://dblp.org/rec/conf/eccv/AtaallahSASZDZSE24) |
| `shalev2025imagerag` | ImageRAG: Dynamic Image Retrieval for Reference-Guided Image Generation | 2026 / proceedings | [record](https://dblp.org/rec/conf/iclr/Shalev-Arkushin26) |
| `arefeen2024irag` | iRAG: Advancing RAG for Videos with an Incremental Approach | 2024 / proceedings | [record](https://dblp.org/rec/conf/cikm/ArefeenDUC24) |
| `lu2022learnexplainmultimodal` | Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering | 2022 / proceedings | [record](https://dblp.org/rec/conf/nips/LuMX0CZTCK22) |
| `liu2023learning` | Learning Customized Visual Models with Retrieval-Augmented Knowledge | 2023 / proceedings | [record](https://dblp.org/rec/conf/cvpr/LiuSY0GLL23) |
| `hu2026longliverag` | LongLive-RAG: A General Retrieval-Augmented Framework for Long Video Generation | 2026 / preprint | [record](https://dblp.org/rec/journals/corr/abs-2606-02553) |
| `cho2024m3docrag` | M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding | 2024 / preprint | [record](https://dblp.org/rec/journals/corr/abs-2411-04952) |
| `chakraborty2026marquis` | MARQUIS: A Three-Stage Pipeline for Video Retrieval-Augmented Generation | 2026 / preprint | [record](https://dblp.org/rec/journals/corr/abs-2605-17640) |
| `mahata2026midr` | MIDR: Enrichment-Augmented Indexing for Multimodal Document Retrieval | 2026 / preprint | [record](https://dblp.org/rec/journals/corr/abs-2609-01316) |
| `chen2024mllm` | MLLM Is a Strong Reranker: Advancing Multimodal Retrieval-augmented Generation via Knowledge-enhanced Reranking and Noise-injected Training | 2024 / preprint | [record](https://dblp.org/rec/journals/corr/abs-2407-21439) |
| `dong2025mmdocir` | MMDocIR: Benchmarking Multimodal Retrieval for Long Documents | 2025 / proceedings | [record](https://dblp.org/rec/conf/emnlp/DongCDLTL25) |
| `xia2024mmed` | MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models | 2025 / proceedings | [record](https://dblp.org/rec/conf/iclr/0005ZLWSWZ0Y25) |
| `hu2025mragbench` | MRAG-Bench: Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models | 2025 / proceedings | [record](https://dblp.org/rec/conf/iclr/0006GDFLC025) |
| `arxiv250524073` | mRAG: Elucidating the Design Space of Multi-modal Retrieval-Augmented Generation | 2025 / preprint | [record](https://dblp.org/rec/journals/corr/abs-2505-24073) |
| `wang2026graphdoc` | Multimodal Graph RAG for Long-range Visually Rich Document Understanding | 2026 / preprint | [record](https://dblp.org/rec/journals/corr/abs-2606-28780) |
| `chen2022murag` | MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text | 2022 / proceedings | [record](https://dblp.org/rec/conf/emnlp/ChenHCVC22) |
| `marino2019okvqa` | OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge | 2019 / proceedings | [record](https://dblp.org/rec/conf/cvpr/MarinoRFM19) |
| `zhang2024omagent` | OmAgent: A Multi-modal Agent Framework for Complex Video Understanding with Task Divide-and-Conquer | 2024 / proceedings | [record](https://dblp.org/rec/conf/emnlp/ZhangZYML24) |
| `hu2023opendomain` | Open-domain Visual Entity Recognition: Towards Recognizing Millions of Wikipedia Entities | 2023 / proceedings | [record](https://dblp.org/rec/conf/iccv/HuLCKJLTC23) |
| `xu2024p` | P-RAG: Progressive Retrieval Augmented Generation For Planning on Embodied Everyday Task | 2024 / proceedings | [record](https://dblp.org/rec/conf/mm/XuWZL24) |
| `wang2024phidias` | Phidias: A Generative Model for Creating 3D Content from Text, Image, and 3D Conditions with Reference-Augmented Diffusion | 2025 / proceedings | [record](https://dblp.org/rec/conf/iclr/00030HH0L25) |
| `mortaheb2025rag` | RAG-Check: Evaluating Multimodal Retrieval Augmented Generation Performance | 2025 / preprint | [record](https://dblp.org/rec/journals/corr/abs-2501-03995) |
| `yuan2024rag` | RAG-Driver: Generalisable Driving Explanations with Retrieval-Augmented In-Context Learning in Multi-Modal Large Language Model | 2024 / preprint | [record](https://dblp.org/rec/journals/corr/abs-2402-10828) |
| `liu2024rar` | RAR: Retrieving and Ranking Augmented MLLMs for Visual Recognition | 2026 / journal | [record](https://dblp.org/rec/journals/tip/LiuSZLZDXLW26) |
| `rao2024raven` | RAVEN: Multitask Retrieval Augmented Vision-Language Learning | 2024 / preprint | [record](https://dblp.org/rec/journals/corr/abs-2406-19150) |
| `wasserman2025realmmra` | REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark | 2025 / proceedings | [record](https://dblp.org/rec/conf/acl/WassermanPNGSBK25) |
| `ding2024realgen` | RealGen: Retrieval Augmented Generation for Controllable Traffic Scenarios | 2024 / proceedings | [record](https://dblp.org/rec/conf/eccv/DingCZXP24) |
| `lyu2025realrag` | RealRAG: Retrieval-augmented Realistic Image Generation via Self-reflective Contrastive Learning | 2025 / proceedings | [record](https://dblp.org/rec/conf/icml/Lyu0JY000H25) |
| `zhang2023remodiffuse` | ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model | 2023 / proceedings | [record](https://dblp.org/rec/conf/iccv/ZhangGPCHLYL23) |
| `lee2026carve` | Rethinking RAG in Long Videos: What to Retrieve and How to Use It? | 2026 / preprint | [record](https://dblp.org/rec/journals/corr/abs-2606-13141) |
| `long2022retrieval` | Retrieval Augmented Classification for Long-Tail Visual Recognition | 2022 / proceedings | [record](https://dblp.org/rec/conf/cvpr/Long0ANPGBSH22) |
| `tan2024retrieval` | Retrieval Meets Reasoning: Even High-school Textbook Knowledge Benefits Multimodal Reasoning | 2024 / preprint | [record](https://dblp.org/rec/journals/corr/abs-2405-20834) |
| `arxiv260304238` | Retrieval or Representation? Reassessing Benchmark Gaps in Multilingual and Visually Rich RAG | 2026 / preprint | [record](https://dblp.org/rec/journals/corr/abs-2603-04238) |
| `blattmann2022retrieval` | Retrieval-Augmented Diffusion Models | 2022 / proceedings | [record](https://dblp.org/rec/conf/nips/BlattmannROMO22) |
| `lewis2020retrieval` | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | 2020 / proceedings | [record](https://dblp.org/rec/conf/nips/LewisPPPKGKLYR020) |
| `gao2023retrieval` | Retrieval-Augmented Generation for Large Language Models: A Survey | 2023 / preprint | [record](https://dblp.org/rec/journals/corr/abs-2312-10997) |
| `ramos2023retrieval` | Retrieval-augmented Image Captioning | 2023 / proceedings | [record](https://dblp.org/rec/conf/eacl/RamosEM23) |
| `horita2024retrieval` | Retrieval-Augmented Layout Transformer for Content-Aware Layout Generation | 2024 / proceedings | [record](https://dblp.org/rec/conf/cvpr/HoritaIKYA24) |
| `kim2024retrieval` | Retrieval-Augmented Open-Vocabulary Object Detection | 2024 / proceedings | [record](https://dblp.org/rec/conf/cvpr/KimCKK24) |
| `seo2024retrieval` | Retrieval-Augmented Score Distillation for Text-to-3D Generation | 2024 / proceedings | [record](https://dblp.org/rec/conf/icml/SeoHJKKLK24) |
| `rao2023retrieval` | Retrieval-based Knowledge Augmented Vision Language Pre-training | 2023 / proceedings | [record](https://dblp.org/rec/conf/mm/RaoSLZY23) |
| `hu2023reveal` | Reveal: Retrieval-Augmented Visual-Language Pre-Training with Multi-Source Multimodal Knowledge Memory | 2023 / proceedings | [record](https://dblp.org/rec/conf/cvpr/HuI0WCSSRF23) |
| `qi2024roravlm` | RoRA-VLM: Robust Retrieval-Augmented Vision Language Models | 2024 / preprint | [record](https://dblp.org/rec/journals/corr/abs-2410-08876) |
| `xia2024rule` | RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models | 2024 / proceedings | [record](https://dblp.org/rec/conf/emnlp/XiaZLZLLZY24) |
| `gao2026scaling` | Scaling Beyond Context: A Survey of Multimodal Retrieval-Augmented Generation for Document Understanding | 2026 / proceedings | [record](https://dblp.org/rec/conf/acl/GaoZJDCCLZBG26) |
| `li2024search` | SearchLVLMs: A Plug-and-Play Framework for Augmenting Large Vision-Language Models by Searching Up-to-Date Internet Knowledge | 2024 / proceedings | [record](https://dblp.org/rec/conf/nips/Li0JLS000Z24) |
| `xiong2025streaming` | Streaming Video Understanding and Multi-round Interaction with Memory-enhanced Knowledge | 2025 / proceedings | [record](https://dblp.org/rec/conf/iclr/XiongYYZ0ZL25) |
| `sankaradas2025streamingragrealtimecontextualretrieval` | StreamingRAG: Real-time Contextual Retrieval and Generation Framework | 2025 / preprint | [record](https://dblp.org/rec/journals/corr/abs-2501-14101) |
| `sun2024surf` | SURf: Teaching Large Vision-Language Models to Selectively Utilize Retrieved Information | 2024 / proceedings | [record](https://dblp.org/rec/conf/emnlp/SunZZSQ024) |
| `wan2024factuality` | The Factuality Tax of Diversity-Intervened Text-to-Image Generation: Benchmark and Fact-Augmented Intervention | 2024 / proceedings | [record](https://dblp.org/rec/conf/emnlp/WanWWC24) |
| `wang2026trajrag` | TrajRAG: Retrieving Geometric-Semantic Experience for Zero-Shot Object Navigation | 2026 / proceedings | [record](https://dblp.org/rec/conf/cvpr/WangZZSDJ26) |
| `arxiv260529956` | Uncertainty Quantification for Multimodal Retrieval Augmented Generation | 2026 / proceedings | [record](https://dblp.org/rec/conf/ictir/BinzSH26) |
| `li2024understanding` | Understanding Retrieval Robustness for Retrieval-augmented Image Captioning | 2024 / proceedings | [record](https://dblp.org/rec/conf/acl/LiLRTE24) |
| `yi2024video` | Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models | 2025 / proceedings | [record](https://dblp.org/rec/conf/cvpr/YiWLNG25) |
| `luo2024video` | Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension | 2025 / proceedings | [record](https://dblp.org/rec/conf/nips/LuoZLYLFHJCLJ25) |
| `jeong2025videorag` | VideoRAG: Retrieval-Augmented Generation over Video Corpus | 2025 / proceedings | [record](https://dblp.org/rec/conf/acl/JeongKBH25) |
| `sun2025visrag2` | VisRAG 2.0: Evidence-Guided Multi-Image Reasoning in Visual Retrieval-Augmented Generation | 2025 / preprint | [record](https://dblp.org/rec/journals/corr/abs-2510-09733) |
| `yu2024visrag` | VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents | 2025 / proceedings | [record](https://dblp.org/rec/conf/iclr/YuTXCRYLWHL025) |
| `wu2025visualrag` | Visual-RAG: Benchmarking Text-to-Image Retrieval Augmented Generation for Visual Knowledge Intensive Queries | 2026 / proceedings | [record](https://dblp.org/rec/conf/sigir/WuLLYW26) |
| `Arefeen2024ViTA` | ViTA: An Efficient Video-to-Text Algorithm using VLM for RAG-based Video Analysis System | 2024 / proceedings | [record](https://dblp.org/rec/conf/cvpr/ArefeenDUC22) |
| `chang2022webqa` | WebQA: Multihop and Multimodal QA | 2022 / proceedings | [record](https://dblp.org/rec/conf/cvpr/ChangCNGSB22) |
| `arxiv260500911` | When Good OCR Is Not Enough: Benchmarking OCR Robustness for Retrieval-Augmented Generation | 2026 / proceedings | [record](https://dblp.org/rec/conf/acl/SunWHZJCZ26) |
