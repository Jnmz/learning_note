# AGENTS

This repository is a long-term technical learning notes and documentation website for AI/ML research.

## Working Rules

- Do not commit or push directly to `main`.
- Before starting work, sync the latest `main`.
- Make all changes, commits, and pushes on `codex-staging`.
- Repository automation promotes `codex-staging` to `main` with a bot identity.
- Agents are not responsible for direct publishing or release handling on `main`.

## Task Completion Report

After each task, report:

- modified files
- commit message
- whether the changes were successfully pushed to `codex-staging`

## Style

- Keep changes small and maintainable.
- Follow existing MkDocs and documentation structure.
- Prefer clear workflow names and minimal automation complexity.

## Technical Learning Note Writing Standards

These rules apply whenever Codex or another AI agent writes or substantially rewrites a technical note in this repository.

### Positioning

- Treat technical notes in this repository as research-style learning notes, not lightweight overviews.
- Write for readers who already have basic machine learning background and want to understand the derivation path, not just the final result.
- Prioritize logical clarity, explicit assumptions, and complete derivations over extreme brevity.

### Default Note Structure

Technical notes should usually include the following sections unless there is a strong reason not to:

- Background / Problem Setup
- Notation
- Core Idea
- Detailed Derivation
- Intuition / Interpretation
- Relation to Other Methods
- My Notes / Open Questions
- References

If a note omits one of these sections, the omission should be intentional rather than accidental.

### Mathematical Derivation Depth

- Do not give only the final formula when the intermediate steps are important to understanding.
- Core formulas must include the intermediate algebraic, probabilistic, or variational steps that connect the starting point to the result.
- Do not rely on vague transitions such as "it is easy to show", "similarly", "one can derive", "it can be proven", or "we directly obtain" unless the skipped step is truly trivial in context.
- When a derivation uses a standard identity, explicitly name it before or during the derivation.
- When introducing a result, state the starting assumptions, definitions, or known equations that the derivation depends on.

For the following recurring derivation types, the note must explicitly state what is being used:

- Bayes' rule
- conditional probability or conditional expectation identities
- Gaussian density formulas and Gaussian quadratic-form manipulations
- reparameterization
- change of variables or variable substitution
- the definition of the score \( \nabla_x \log p(x) \)
- the relationship between the continuity equation, ODEs, and SDEs when relevant

### Derivation Block Requirements

- Every theory-heavy note should contain at least 2 to 3 complete derivation blocks.
- A derivation block should start from a definition, target objective, known formula, or modeling assumption and proceed step by step to the conclusion.
- Do not jump from one displayed equation to another with only a phrase like "thus" or "from the above" when the intermediate logic is nontrivial.
- If a derivation is too long for the main flow, split it into labeled substeps rather than collapsing it into a single sentence.
- When simplifications depend on a specific approximation, independence assumption, Gaussian assumption, or boundary condition, state that explicitly.

### Intuition And Cross-Method Context

- After detailed derivations, explain what the result means geometrically, probabilistically, or algorithmically.
- When a note covers a method that is closely related to other families, include a short section clarifying the relationship rather than leaving formulas isolated.
- Prefer explanations that connect objectives, parameterizations, and sampling or inference procedures.

### Diagram Use

- Add a simple diagram when the concept is abstract, the path or dependency structure is hard to visualize, or the training and sampling flows are easy to confuse.
- Diagrams should support understanding rather than decoration.
- Prefer diagrams such as process flows, relation maps, probability paths, or variable dependency graphs.
- Keep diagrams simple enough that they can be maintained alongside the Markdown note.

### Writing Style Constraints

- Do not default to a survey-style summary if the topic is fundamentally mathematical or derivational.
- Avoid compressing a difficult derivation into a short paragraph just to keep the note brief.
- Define notation before using it heavily.
- Distinguish clearly between formal derivation, intuition, and personal commentary.
- End with concrete references that readers can use to trace the original sources.

## Math Rendering And Build Rules

These rules apply whenever an agent adds or edits mathematical content in this repository.

- This repository uses MkDocs + Material + arithmatex + MathJax. Mathematical content must not only build successfully, but also render correctly in the generated site.
- In Markdown list items, avoid placing block-level display equations directly under a bullet unless the current rendering path is known to support that structure correctly.
- For mathematical content, prefer safer rendering patterns:
  - inline math for short expressions inside list items or prose
  - standalone block equations outside lists
  - plain paragraph splits when mixing explanation and formulas is clearer
- After adding or modifying a technical note that contains mathematical formulas, the agent must run at least one site build check.
- If the change affects important formula presentation, the agent should actively verify rendering behavior and not treat `mkdocs build` success alone as sufficient evidence.
- If a rendering issue appears, prefer rewriting the Markdown structure into a more stable form before changing global site configuration.

## Staging To Main Promotion Checks

These rules apply after changes are pushed to `codex-staging`.

- Task completion should not stop at "pushed to codex-staging" when the repository automation is expected to promote changes to `main`.
- After pushing to `codex-staging`, the agent should try to check:
  - whether the push succeeded
  - whether the relevant workflow was triggered
  - whether promotion to `main` appears to have completed
- If promotion fails, the agent should first try the obvious repository-maintenance steps:
  - sync the latest `origin/main` into `codex-staging`
  - resolve straightforward text or documentation conflicts
  - push the updated `codex-staging` branch again
- If the workflow still does not close the loop automatically, the agent must report clearly:
  - which step failed
  - whether the issue is a workflow failure, branch conflict, rendering problem, or permission problem
  - what manual action is still required
- Final task reports should include not only whether the changes were pushed to `codex-staging`, but also whether the agent observed the changes reaching `main`; if not, the report should state the reason.
