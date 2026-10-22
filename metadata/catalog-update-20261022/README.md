# MultiHaystack catalog addition — 22 October 2026

Add one benchmark under `evaluation`: MultiHaystack, ECCV 2026. The formal [DBLP record](https://dblp.org/rec/conf/eccv/XuYCYHSGPF26) and its [preprint identity](https://dblp.org/rec/journals/corr/abs-2603-05697) refer to the same title and author list, so the catalog counts them once.

## Bibliographic evidence

- The [DBLP proceedings table of contents](https://dblp.dagstuhl.de/db/conf/eccv/eccv2026-40.html) lists the complete title, nine ordered authors, the formal record key and pages 286–304 in ECCV 2026, Part XL.
- The [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-032-37321-2_16) independently establishes the title, authors and formal publication. Springer records first online publication on 22 September 2026.
- The [primary manuscript](https://arxiv.org/abs/2603.05697) describes 747 evidence-grounded questions and more than 46,000 retrieval candidates across images, videos and documents.
- The [author project page](https://danielxu0208.github.io/MultiHaystack.github.io/) links the [official code](https://github.com/Danielxu0208/MultiHaystack) and [dataset](https://huggingface.co/datasets/DanielXu0208/MultiHaystack_Full).

`multihaystack-evidence.json` is an explicitly labeled manual transcription of these bibliographic facts. It is an equivalent evidence record, not a raw network or SPARQL response. Its `results.bindings` representation preserves compatibility with the existing validator.

## Scope and validation

MultiHaystack evaluates evidence retrieval from a heterogeneous external corpus and downstream question answering together. Its primary category is `evaluation` and its role is `Benchmark`. New manuscript citation keys and legacy row indices are empty. The fixed manuscript audits remain historical snapshots.

Regenerate and validate the source with:

```sh
python3 scripts/render_catalog.py
python3 scripts/render_catalog.py --check
```
