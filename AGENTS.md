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
