# Learning Notes

A Markdown-first technical notes site for long-term AI/ML research learning.

This repository is scaffolded as a documentation website using MkDocs + Material for MkDocs. It is optimized for:

- hierarchical topic organization
- lightweight Markdown authoring
- internal linking and navigable indexes
- consistent note structure for future expansion
- straightforward GitHub Pages publishing

## Structure

```text
.
├── AGENTS.md
├── .github/workflows/codex-apply.yml
├── .github/workflows/codex-promote.yml
├── .github/workflows/docs-update.yml
├── .github/workflows/deploy.yml
├── docs/
│   ├── index.md
│   ├── contributing.md
│   ├── workflows/note-template.md
│   ├── unified-models/
│   ├── vlm/
│   ├── generative-modeling/
│   └── stylesheets/extra.css
├── mkdocs.yml
└── requirements.txt
```

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve
```

Then open `http://127.0.0.1:8000`.

## Adding New Notes

1. Copy `docs/workflows/note-template.md` into the appropriate topic folder.
2. Rename the file using a short kebab-case title, for example `mixture-of-experts-notes.md`.
3. Fill in the template sections.
4. Add a link to the new note from that topic's `index.md`.
5. Add the note to `nav` in `mkdocs.yml` if you want it surfaced in the top navigation.

## Publishing

GitHub Pages deployment is prepared through `.github/workflows/deploy.yml`.

To publish:

1. Push this repository to GitHub.
2. In the repository settings, open `Settings -> Pages`.
3. Set `Source` to `GitHub Actions`.
4. Push to the default branch; the workflow will build and deploy the site.

## Automation

Daily AI or Codex changes should go to `codex-staging` first. Prefer `.github/workflows/codex-apply.yml` for automated write-back to `codex-staging`, and let `.github/workflows/codex-promote.yml` synchronize those changes to `main` with the configured bot identity. `.github/workflows/docs-update.yml` is the lightweight companion check for docs, MkDocs config, and README changes, and does not replace the existing Pages deploy workflow.

## Notes Workflow

Use `docs/contributing.md` as the canonical note-writing workflow and `docs/workflows/note-template.md` as the reusable template.
