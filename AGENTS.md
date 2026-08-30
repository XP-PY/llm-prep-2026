# Repository Guidelines

## Project Structure & Module Organization
This repository is documentation-first. Topic notes live under `docs/`, grouped by area such as `docs/Preference_Alignment/`, `docs/Attention_Machanisms/`, and `docs/Inference_Optimization/`. Companion notebooks live in `src/` as `part*.ipynb` files for worked examples and experiments. `README.md` is the main index; update it when adding, renaming, or reorganizing major topics.

## Build, Test, and Development Commands
There is no formal build system or CI test suite configured today. Use lightweight checks while editing:

- `git status` to confirm the intended file set.
- `rg "GRPO|PPO" docs README.md` to find references before renaming or cross-linking topics.
- `python -m json.tool < src/part2_gqa.ipynb >/dev/null` to sanity-check notebook JSON after edits.

For Markdown changes, preview rendering locally and verify relative links still resolve.

## Coding Style & Naming Conventions
Write concise, instructional Markdown. Prefer short sections, direct explanations, and repository-relative links such as `./docs/Preference_Alignment/GRPO.md`. Match existing naming patterns:

- topic docs: mixed case with underscores, e.g. `DeepSeek_V3.md`, `Specialized_LoRA.md`
- notebooks: `src/partN_topic.ipynb`

Keep terminology consistent with neighboring documents, especially for alignment, inference, and architecture topics.

## Research Note Requirements

### Paper Notes

- Use the paper PDF specified by the user as the primary source. Do not substitute a web summary for the supplied paper.
- Save the note in the appropriate model or topic directory. Current model-note locations include:
  - robot policies: `docs/Model_Zoo/Robotics/Policies/`
  - vision models: `docs/Model_Zoo/Vision_Models/`
  - vision-language models: `docs/Model_Zoo/Vision_Language_Models/`
- Match the structure, terminology, and level of detail of neighboring notes in the destination directory.
- Begin with a brief description that identifies the model or method and its main contribution.
- Explain the central method precisely. Use mathematical notation when it makes the architecture, objective, training process, or inference procedure clearer.
- Cover the aspects requested for the specific paper, such as policy structure, datasets, training settings, inference, ablations, and evaluation results. Clearly distinguish pre-training, post-training, fine-tuning, and zero-shot evaluation where applicable.
- Keep the explanation concise and avoid duplicated sections or extensive copying from the paper.
- Add only figures that materially improve understanding. Crop them from the paper, store them under `assets/<Note_Name>/`, reference them with repository-relative paths, and add a short descriptive caption.
- Update the relevant `README.md` index and file tree when adding, renaming, moving, or removing a paper note.

### Book Notes

- Base each note directly on the book PDF supplied by the user.
- Build book notes incrementally, one requested chapter at a time. Append new chapters to the existing book note rather than creating one file per chapter.
- Keep a navigable catalog or table of contents at the top of each book note and update it whenever a chapter is added.
- Do not extensively copy the chapter. Explain its core material precisely, prioritizing important definitions, formulas, algorithms, and short clarifying examples.
- Add figures from the book only when they are necessary for a clear explanation. Store extracted figures under the corresponding book asset directory and provide descriptive captions.
- Maintain consistent notation across chapters and with the source book. Check that equations, Mermaid diagrams, code fences, anchors, and relative links render correctly.
- Current book-note locations are:
  - robotics: `docs/Books/Robotics/Modern_Robotics.md`
  - reinforcement learning: `docs/Books/Reinforcement_Learning/Reinforcement_Learning_An_Introduction.md`

## Testing Guidelines
Testing here is mostly content validation:

- check Markdown links after edits
- confirm equations and code fences render correctly
- ensure renamed files are reflected in `README.md`
- open edited notebooks to verify they still load

If you add runnable code or scripts later, document the execution command in the relevant note.

## Commit & Pull Request Guidelines
Recent commits use short, imperative messages such as `Updated README` and `Completed DeepSeek-V3.2`. Follow that style, for example `Updated GRPO explanation` or `Completed PPO notes`.

Pull requests should include a brief summary, affected paths, and any screenshots only when notebook output or rendered diagrams materially change. Link related issues or discussion threads when relevant.

## Contributor Notes
Avoid casual directory reshuffles inside `docs/`; cross-links and the README index depend on the current taxonomy. When expanding a subject, prefer extending the existing document before creating a near-duplicate note.
