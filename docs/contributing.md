# Contribution Guide

This repository is intended to stay consistent as it grows. Use the workflow below whenever you add a new note.

## Recommended Workflow

1. Pick the best topic folder for the note.
2. Copy the [note template](./workflows/note-template.md).
3. Save the new file with a descriptive kebab-case filename.
4. Fill in the front section first: title, scope, references, and key takeaway.
5. Add the note to the corresponding topic index page.
6. If the note should appear in top-level navigation, update `mkdocs.yml`.
7. Preview locally with `mkdocs serve` before publishing.

## Writing Conventions

- One note should cover one paper, concept, model family, or tightly related cluster of ideas.
- Prefer explanatory prose over raw bullet dumps.
- Use short sections with informative headings.
- Include citations, links, or paper identifiers when available.
- End with open questions or follow-up directions when useful.

## File Naming

Use lowercase kebab-case:

```text
scaling-laws-notes.md
segment-anything-overview.md
multimodal-tokenization.md
```

## Topic Index Maintenance

Each topic folder should include an `index.md` that acts as the landing page for that section. Whenever you add a note:

- add a one-line description and link in that topic index
- keep notes grouped logically
- update `mkdocs.yml` when prominent navigation should change

## Internal Linking

- Link related notes directly using relative Markdown links.
- Prefer stable links to topic indexes for broad navigation.
- Add "See also" sections when concepts connect across topics.

## Minimum Note Checklist

- Clear title
- Short summary
- Context or motivation
- Core ideas or mechanisms
- References
- Open questions, caveats, or follow-ups
