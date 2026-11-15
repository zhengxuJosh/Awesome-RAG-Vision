#!/usr/bin/env python3
"""Validate papers.json and deterministically render the public catalog (stdlib only)."""
import argparse
import collections
import csv
import hashlib
import io
import json
import re
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = [
    ('surveys', 'Surveys and overviews'),
    ('evaluation', 'Benchmarks, datasets and evaluation'),
    ('understanding', 'Visual understanding and multimodal reasoning'),
    ('documents', 'Document understanding'),
    ('video', 'Video understanding'),
    ('medical', 'Medical vision and clinical documents'),
    ('synthesis', 'Image and video generation'),
    ('3d', '3D scenes and motion generation'),
    ('embodied', 'Embodied memory, navigation and driving'),
    ('applications', 'Domain applications'),
    ('retrieval', 'Retrieval and reranking components'),
    ('background', 'Foundations and related techniques'),
]
ROLES = {'Method', 'Component', 'Benchmark', 'Analysis', 'Survey', 'Background', 'Shared-task report'}


def validate(data):
    assert re.fullmatch(r'\d{4}-\d{2}-\d{2}', data['updated'])
    seen_ids, seen_titles, seen_records = set(), set(), set()
    categories = dict(CATEGORIES)
    for p in data['papers']:
        assert p['id'] not in seen_ids, f"Duplicate ID: {p['id']}"
        seen_ids.add(p['id'])
        title = re.sub(r'\W', '', p['title'].casefold())
        assert title and title not in seen_titles, f"Duplicate title: {p['title']}"
        seen_titles.add(title)
        assert p['category'] in categories and p['role'] in ROLES
        assert p['publication_status'] in {'proceedings', 'journal', 'preprint'}
        assert p['authors']
        assert all(isinstance(a, str) and a.strip() for a in p['authors'])
        assert 1900 < p['year'] <= int(data['updated'][:4])
        assert isinstance(p['publication_year'], int)
        assert p['dblp_url'] == 'https://dblp.org/rec/' + p['id']
        assert p['dblp_url'] in p['dblp_records']
        for record in p['dblp_records']:
            assert record not in seen_records, f"DBLP version assigned twice: {record}"
            seen_records.add(record)
        for field in ['paper_url', 'dblp_url', 'code_url', 'code_source', 'publication_source']:
            if field in p:
                parsed = urlparse(p[field])
                assert parsed.scheme in {'http', 'https'} and parsed.netloc, (p['id'], field)
        assert not p.get('code_url') or p.get('code_source'), 'Code requires a provenance URL'
        if p['publication_status'] != 'preprint' and '/journals/corr/' in p['dblp_url']:
            assert p.get('publication_source'), 'Formal venue needs proceedings evidence'
        assert p['verified_on'] <= data['updated']
    # The repository update must preserve every citation in the fixed manuscript snapshot.
    snapshot = json.loads((ROOT / 'metadata/dblp-audit-20261015/verified-records.json').read_text())
    assert {r['citation_key'] for r in snapshot} == {
        k for p in data['papers'] for k in p['manuscript_citation_keys']
    }, 'Manuscript subset lost during catalog maintenance'


def validate_evidence(data):
    """Match each catalog identity and author sequence to saved DBLP responses."""
    snapshot = json.loads((ROOT / 'metadata/dblp-audit-20261015/verified-records.json').read_text())
    titles = {r['dblp_url']: {t.rstrip('.') for t in r['properties']['title']} for r in snapshot}
    names = {r['dblp_url']: {int(a['ordinal']): a['name'] for a in r['authors']} for r in snapshot}
    for meta in sorted((ROOT / 'metadata').glob('catalog-update-*')):
        fresh_names = {}
        for path in sorted(meta.glob('*.json')):
            response = json.loads(path.read_text())
            if not isinstance(response, dict):
                continue
            for row in response.get('results', {}).get('bindings', []):
                if 'property' in row and row['property']['value'].endswith('#title'):
                    titles.setdefault(row['p']['value'], set()).add(row['value']['value'].rstrip('.'))
                if 'name' in row and 'ordinal' in row and 'p' in row:
                    record = row['p']['value']
                    ordinal = int(row['ordinal']['value'])
                    name = row['name']['value']
                    previous = fresh_names.setdefault(record, {}).get(ordinal)
                    assert previous in {None, name}, f'Conflicting author signature: {record}'
                    fresh_names[record][ordinal] = name
        names.update(fresh_names)
    for p in data['papers']:
        record = p['dblp_url']
        assert p['title'] in titles.get(record, set()), f'Title does not match source: {record}'
        source_names = [re.sub(r' \d{4}$', '', name) for _, name in sorted(names[record].items())]
        assert source_names == p['authors'], f'Authors do not match source: {record}'
        assert all(version in titles for version in p['dblp_records']), f'Unverified version: {record}'
        assert set(names[record]) == set(range(1, len(source_names) + 1)), f'Missing author ordinal: {record}'


def key(p):
    if p['id'] == 'journals/corr/abs-2503-18016':
        return 'zheng2025retrievalvision'
    return 'visionrag' + str(p['year']) + hashlib.sha1(p['id'].encode()).hexdigest()[:10]


def tex(s):
    return s.replace('\\', r'\textbackslash{}').replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('#', r'\#')


def bib(p):
    kind = 'inproceedings' if p['publication_status'] == 'proceedings' else 'article'
    venue_field = 'booktitle' if kind == 'inproceedings' else 'journal'
    lines = [f'@{kind}{{{key(p)},',
             '  title = {{' + tex(p['title']) + '}},',
             '  author = {' + ' and '.join(tex(a) for a in p['authors']) + '},',
             f'  {venue_field} = {{{tex(p["venue"])}}},',
             f'  year = {{{p["year"]}}},',
             f'  url = {{{p["paper_url"]}}},',
             f'  biburl = {{{p["dblp_url"]}}}']
    if p['publication_status'] == 'preprint':
        lines[-1] += ','
        lines.append('  note = {Preprint; not a peer-reviewed venue claim}')
    return '\n'.join(lines + ['}'])


def md(s):
    return s.replace('|', '&#124;').replace('[', '&#91;').replace(']', '&#93;')


def row(p):
    name = p['authors'][0] + (' et al.' if len(p['authors']) > 1 else '')
    links = f'[DBLP]({p["dblp_url"]})'
    if p.get('code_url'):
        links += f' · [Code]({p["code_url"]})'
    if p['preprint_urls'] and p['preprint_urls'][0] != p['paper_url']:
        links += f' · [arXiv]({p["preprint_urls"][0]})'
    note = f'<br><sub>{md(p["note"])}</sub>' if p.get('note') else ''
    return (f'| [{md(p["title"])}]({p["paper_url"]})<br><sub>{md(name)}</sub>{note}'
            f' | {md(p["venue"])} {p["year"]} | {p["role"]} | {links} |')


def render(data):
    papers = data['papers']
    counts = collections.Counter(p['category'] for p in papers)
    status = collections.Counter(p['publication_status'] for p in papers)
    formal = status['proceedings'] + status['journal']
    readme = f'''<!-- Generated by scripts/render_catalog.py from data/papers.json. -->
<div align="center">

# Awesome RAG in Computer Vision

**Retrieval-augmented understanding, generation and embodied intelligence**

![Papers](https://img.shields.io/badge/papers-{len(papers)}-405B76)
![DBLP verified](https://img.shields.io/badge/identity-DBLP%20verified-50766B)
![Updated](https://img.shields.io/badge/updated-{data['updated']}-777777)

[Survey](https://arxiv.org/abs/2503.18016) · [Paper list](#paper-list) · [JSON](data/papers.json) · [CSV](data/papers.csv) · [BibTeX](data/papers.bib) · [Contribute](CONTRIBUTING.md)

</div>

A curated research guide to **how retrieved evidence, exemplars, visual references and memory support vision models**. Covers images, documents, video, medical applications, visual synthesis, 3D and embodied agents.

**{len(papers)} unique papers · {formal} proceedings/journal papers · {status['preprint']} preprints · 12 categories.** Every entry has a direct DBLP record. A linked proceedings page supplies formal venue evidence where only the preprint is indexed in DBLP. Conference and preprint versions count once.

<p align="center"><img src="assets/acl-figures/overview.png" alt="Four roles of retrieved information: evidence, exemplars, synthesis references and embodied memory" width="100%"></p>

*From the survey revision: four roles of retrieved information. [More figures and editable sources](FIGURES.md).*

## Start reading

These are entry points for different questions, rather than a performance ranking.

| Your question | Suggested entry points | What to look for |
|---|---|---|
'''
    routes = [
        ('How does multimodal RAG differ from text-only RAG?', ['MuRAG:', 'UniversalRAG:'], 'Modalities and granularity of the retrieved evidence'),
        ('How do we retrieve document pages?', ['VisRAG:', 'ColPali:'], 'Separate the downstream answer pipeline from its retrieval component'),
        ('How is video evidence selected?', ['VideoRAG:', 'Video-RAG:'], 'Retrieval over a video corpus versus a supplied video'),
        ('How do we evaluate visual retrieval augmentation?', ['MRAG-Bench:', 'REAL-MM-RAG:'], 'Candidate collections, relevance labels and answer evaluation'),
        ('How can retrieval guide visual synthesis?', ['Re-Imagen:', 'ImageRAG:'], 'Retrieved visual references and generation conditioning'),
        ('How does agent memory support actions?', ['Embodied-RAG:', 'TrajRAG:'], 'Memory construction, retrieval and the downstream action'),
        ('What changes in medical applications?', ['MMed-RAG:', 'RULE:'], 'Clinical evidence, reliability and domain-specific retrieval'),
    ]
    for question, prefixes, focus in routes:
        links = []
        for prefix in prefixes:
            p = next(p for p in papers if p['title'].startswith(prefix))
            label = prefix.rstrip(':')
            links.append(f'[{label}]({p["paper_url"]})')
        readme += f'| {question} | {" · ".join(links)} | {focus} |\n'
    updated_date = data['updated']
    readme += f'''
## Recent verified additions and publication updates

Targeted update: **{updated_date}**. The full list below is sorted by publication/event year, then title. This selection highlights additions and metadata changes; it is not an exhaustive search of all recent work.

| Paper | Verified publication | Why it is useful here |
|---|---|---|
'''
    recent = [
        ('Seeing Through the MiRAGE:', 'Subclaim-level information quality and citation support; formal EMNLP publication'),
        ('Doc-REFRAG:', 'Question-guided chunk expansion after document retrieval'),
        ('UNIDOC-BENCH:', 'Formal Findings of EMNLP publication; shared-pool document evaluation'),
        ('UniversalRAG:', 'Unified retrieval across modalities and granularities; merged with its differently titled preprint'),
        ('LVLM-Aware', 'Medical retrieval with general-purpose vision-language models'),
        ('Region-R1:', 'Query-side cropping for reranking; classified as a component'),
        ('RAVENEA:', 'Visual culture understanding benchmark'),
        ('MKG-RAG-Bench:', 'Retrieval evaluation for multimodal knowledge graphs'),
        ('M3Retrieve:', 'Multimodal retrieval benchmark for medicine'),
        ('CRAFT:', 'Adaptive key-frame selection for video QA'),
        ('MARQUIS:', 'A staged video retrieval-augmented pipeline'),
        ('TRACE:', 'Evidence-grounded understanding across multiple videos'),
        ('Bridging the Semantic-Utility', 'Generator feedback for multimodal retrieval alignment; preprint'),
        ('Complementary Retrieval-Augmented', 'Long-form video generation; formerly pending, now DBLP matched'),
    ]
    for prefix, why in recent:
        p = next(p for p in papers if p['title'].startswith(prefix))
        readme += f'| [{md(p["title"])}]({p["paper_url"]}) | {p["venue"]} {p["year"]} · [DBLP]({p["dblp_url"]}) | {why} |\n'
    readme += '''
<a id="paper-list"></a>

## Paper list

**Roles:** `Method` covers retrieval-augmented understanding, generation or agent methods; it does not imply every paper provides a deployable end-to-end RAG system. `Component` denotes a retriever, indexer or reranker. `Benchmark` includes datasets and evaluation collections. `Analysis`, `Shared-task report`, `Survey` and `Background` are counted separately in the data.

Use GitHub's page search (**Ctrl/Cmd + F**) to find a title or method. For filtering by year, venue, role or topic, download the [CSV](data/papers.csv) or [JSON](data/papers.json). Author lists in the exports are complete and ordered as in DBLP.

| Topic | Unique papers |
|---|---:|
'''
    for cat, title in CATEGORIES:
        readme += f'| [{title}](#{cat}) | {counts[cat]} |\n'
    for cat, title in CATEGORIES:
        readme += f'\n<a id="{cat}"></a>\n\n### {title}\n\n'
        if cat == 'evaluation':
            readme += 'Includes general VQA/document datasets used in RAG studies; inclusion does not mean a dataset was designed specifically for RAG.\n\n'
        if cat == 'background':
            readme += 'Related techniques and cross-domain examples. These are excluded from the visual retrieval-augmentation method count.\n\n'
        readme += '| Paper / first author | Publication | Role | Resources |\n|---|---|---|---|\n'
        for p in sorted((p for p in papers if p['category'] == cat), key=lambda p: (-p['year'], p['title'].casefold())):
            readme += row(p) + '\n'
    readme += '''
## Scope and verification

We use **retrieval augmentation** as an umbrella for retrieval-augmented understanding and generation. For each method, distinguish its candidate collection, query, retrieved unit and downstream use. External-corpus retrieval, retrieval within a supplied document/video, and retrieval from accumulated agent memory are different settings.

The catalog records **bibliographic identity**, rather than certifying experimental claims, reproducibility or a complete full-text review. Formal venue/year is taken from DBLP or an explicitly linked proceedings page; an arXiv acceptance claim alone is insufficient. The conference event year is preferred when its volume has a later imprint year. Preprints are explicitly labeled `arXiv`.

- [16 November additions and publication updates](metadata/catalog-update-20261116/README.md)
- [October catalog verification, version merges, corrections and review queue](metadata/catalog-update-20261019/README.md)
- [Fixed 81-reference manuscript bibliography audit](VERIFIED_REFERENCES.md)
- [21-study analytical extraction and eight supporting full texts](metadata/synthesis-audit-20261018/README.md)
- [Previous homepage snapshot](archive/README-before-20261019.md)

The public catalog is broader than the fixed manuscript bibliography. Adding a paper here does **not** mean it has been added to the manuscript or analyzed in full. Nine author-only survey entries from the old homepage remain in the review queue and are excluded from the verified count.

## Resources and contributions

- [Workshops and implementation tutorials](RESOURCES.md): distinct from peer-reviewed research evidence.
- [Add or correct a paper](CONTRIBUTING.md): provide the full title, direct DBLP/Scholar record, primary publication link and a justified category.
- [Report a missing paper](https://github.com/zhengxuJosh/Awesome-RAG-Vision/issues/new?template=paper.yml).

The homepage and exports are generated from `data/papers.json`. Run `python3 scripts/render_catalog.py` after editing the catalog, and `python3 scripts/render_catalog.py --check` before opening a pull request.

## Citation

If this collection helps your research, please cite the survey:

```bibtex
'''
    own = next(p for p in papers if p['id'] == 'journals/corr/abs-2503-18016')
    readme += bib(own) + '\n```\n'
    csvfile = io.StringIO(newline='')
    columns = ['id', 'title', 'authors', 'year', 'publication_year', 'venue', 'publication_status', 'category', 'role', 'paper_url', 'dblp_url', 'code_url', 'verified_on', 'note']
    writer = csv.DictWriter(csvfile, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    for p in papers:
        writer.writerow({col: '; '.join(p[col]) if col == 'authors' else p.get(col, '') for col in columns})
    return {'README.md': readme, 'data/papers.csv': csvfile.getvalue(),
            'data/papers.bib': '% Generated from data/papers.json; UTF-8.\n\n' + '\n\n'.join(bib(p) for p in papers) + '\n'}


def check_links(readme, generated_files=()):
    anchors = set(re.findall(r'<a id="([^"]+)">', readme))
    for heading in re.findall(r'^#{1,6} (.+)$', readme, re.M):
        anchors.add(re.sub(r'[^\w\- ]', '', heading.lower()).replace(' ', '-'))
    for link in re.findall(r'\]\(([^)]+)\)', readme):
        if link.startswith(('https://', 'http://')):
            continue
        if link.startswith('#'):
            assert link[1:] in anchors, f'Missing anchor: {link}'
        else:
            assert link in generated_files or (ROOT / link).is_file(), f'Missing local link: {link}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Fail if generated files are stale')
    args = parser.parse_args()
    data = json.loads((ROOT / 'data/papers.json').read_text())
    validate(data)
    validate_evidence(data)
    outputs = render(data)
    check_links(outputs['README.md'], outputs)
    stale = []
    for filename, content in outputs.items():
        path = ROOT / filename
        if args.check:
            if not path.exists() or path.read_text() != content:
                stale.append(filename)
        else:
            path.write_text(content)
    if stale:
        raise SystemExit('Stale generated files: ' + ', '.join(stale))
    print(f'{len(data["papers"])} unique papers validated; ' + ('generated files current.' if args.check else 'homepage and exports generated.'))


if __name__ == '__main__':
    main()
