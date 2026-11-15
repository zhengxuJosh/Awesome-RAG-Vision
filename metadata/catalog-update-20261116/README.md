# Targeted catalog update — 16 November 2026

The catalog now contains **177 unique DBLP-matched papers**. This update adds **MiRAGE** and **Doc-REFRAG** and updates **UniDoc-Bench** to its formal Findings of EMNLP 2026 publication. Exact-title/DOI responses and ordered author signatures are saved here. MiRAGE and UniDoc-Bench use DBLP preprint identities matched by title and ordered authors to the formal proceedings metadata. The November manuscript addendum records these three papers separately from its fixed October analytical extraction.

- [MiRAGE](https://aclanthology.org/2026.emnlp-main.1765/): information factuality/coverage and citation support/completeness at the subclaim level.
- [Doc-REFRAG](https://aclanthology.org/2026.emnlp-main.377/): question-conditioned expansion of compressed visual chunks after document retrieval.
- [UniDoc-Bench](https://aclanthology.org/2026.findings-emnlp.534/): document-centric multimodal retrieval/answer evaluation over aligned candidate pools.

Relevant official source pages were visually inspected; no new quantitative comparison or full-paper configuration audit is claimed. UniDoc-Bench's abstract and PDF Section 3.3 differ on the scope of human verification. No reconciled annotation coverage rate is asserted. `manuscript-addendum.json` saves source hashes, inspected locators and inference limits, without redistributing PDFs.

This is a targeted update, not an exhaustive search through November. Two official-publication candidates (E2E-VisRAG and Retrieval-Induced Mirage) were deferred because this pass did not establish a direct DBLP or Scholar identity record. The nine older unresolved author-only entries remain in the October review queue; they are not counted in the verified catalog.

`manuscript_citation_keys` continues to map the fixed October bibliography. The optional `manuscript_update_citation_key` records November additions without rewriting that historical subset. Run `python3 scripts/render_catalog.py --check` to check the current catalog against saved database responses.
