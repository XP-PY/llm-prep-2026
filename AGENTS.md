# Repository Guidelines

## Project Structure & Module Organization
This repository is documentation-first. `docs/` has two topic roots: `docs/Foundations/` for reusable concepts, engineering techniques, and book notes, and `docs/Model_Zoo/` for studies of specific models, algorithms, policies, systems, and benchmarks. Foundation topics include `Attention_Mechanisms/`, `Preference_Alignment/`, and `Inference_Optimization/`; book notes live in `docs/Foundations/Books/`. Companion notebooks live in `src/` as `part*.ipynb` files for worked examples and experiments. `README.md` is the main index; update it when adding, renaming, or reorganizing major topics.

## Build, Test, and Development Commands
There is no formal build system or CI test suite configured today. Use lightweight checks while editing:

- `git status` to confirm the intended file set.
- `rg "GRPO|PPO" docs README.md` to find references before renaming or cross-linking topics.
- `python -m json.tool < src/part2_gqa.ipynb >/dev/null` to sanity-check notebook JSON after edits.

For Markdown changes, preview rendering locally and verify relative links still resolve.

## Coding Style & Naming Conventions
Write concise, instructional Markdown. Prefer short sections, direct explanations, and repository-relative links such as `./docs/Foundations/Preference_Alignment/GRPO.md`. Match existing naming patterns:

- topic docs: mixed case with underscores, e.g. `DeepSeek_V3.md`, `Specialized_LoRA.md`
- notebooks: `src/partN_topic.ipynb`

Keep terminology consistent with neighboring documents, especially for alignment, inference, and architecture topics.

Keep new notes within the two existing `docs/` roots. Use `Foundations/` when the note primarily explains a reusable concept or technique, and `Model_Zoo/` when it primarily studies a specific model, method, system, or benchmark. Classify by the note's purpose rather than whether its source is a paper or a book. Preserve the directory spellings `Attention_Mechanisms/` and `Position_Embeddings/`.

## Research Note Requirements

- Treat `tmp/` as temporary raw-source input only. Do not link to it or reference its filenames in finished notes. Cite stable public paper/book URLs, DOIs, or official project pages instead; if none is available, provide bibliographic details without a local-source link.
- Make the logical relationships between concepts explicit in both paper and book notes. Organize explanations around a progression of questions or a derivation, not a list of topics: explain why each concept is introduced, which earlier result it depends on, and how it enables the next step. Connect sections with substantive transitions and, where useful, carry one worked example through the progression. A roadmap alone does not replace these connections within the explanation.

### Paper Notes

- Use the paper PDF specified by the user as the primary source. Do not substitute a web summary for the supplied paper.
- The primary goal is to make the paper easy to understand, not merely to record its contents. Write for a reader with relevant background who has not read the paper. The note should let that reader explain the problem, follow the method, and interpret the evidence without repeatedly reopening the PDF.
- Save the note in the appropriate model or topic directory. Current model-note locations include:
  - robot policies: `docs/Model_Zoo/Robotics/Policies/`
  - agentic robotics systems: `docs/Model_Zoo/Robotics/Agentic_Systems/`
  - vision models: `docs/Model_Zoo/Vision_Models/`
  - vision-language models: `docs/Model_Zoo/Vision_Language_Models/`
  - reinforcement learning algorithms: `docs/Model_Zoo/Reinforcement_Learning/`
- Match the organization and terminology of neighboring notes in the destination directory. Reader comprehension takes priority over copying their brevity or unexplained shorthand.
- Begin with a brief description that identifies the model or method and its main contribution.

#### Explain the Method

- Explain the cause of a problem before presenting the proposed fix. Supply the small amount of background needed to understand that cause, even when it comes from an established subject. For example, explaining accumulated robot commands requires first explaining why a commanded angle can differ from the measured angle and why that difference can supply holding torque. Naming the distinction does not explain it.
- For each central design choice, connect **the concrete difficulty → why the simpler approach fails → what the proposed component changes → why that change helps → how its result feeds the next step**. Identify inputs, relevant state, intermediate operations, decision conditions, and outputs. A list of components, a roadmap, or a claim that something “improves precision” does not supply this causal explanation.
- Lead sections with a plain-language statement of the question and main idea. Introduce one reasoning step at a time; put secondary thresholds, configuration values, and implementation details after the reader understands the mechanism. A technical heading must not carry reasoning that the body leaves unexplained.
- Define paper-specific terms, abbreviations, labels, and symbols at first use through what they mean operationally. Explicitly distinguish concepts that can be confused, such as a command versus a measurement, position versus orientation, input feedback versus output scaling, and an action finishing versus its outcome being checked. Show a situation in which the distinction changes the outcome.
- Demonstrate each central, non-obvious mechanism with a minimal worked example. State the starting conditions, walk through the actual calculation or decisions, and interpret the result. For an update rule, show the state before and after; for a representation, show what happens when the represented quantity changes; for a decision rule, show which branch is taken and why. Reuse an example across related components when useful. Merely saying “suppose the robot needs precision” or naming an application is not a worked example.
- Introduce intuition before equations, then explain how the equation implements it. Define symbols, units, coordinate frames, dimensions, and assumptions where relevant; explain the important operations and substitute simple values when they reveal the mechanism. Distinguish the quantity computed from its downstream physical or computational effect, such as a reference increment versus actual joint motion. A correct formula plus a symbol glossary is insufficient.
- Explain the logic of aggregation, gating, clipping, and scaling rules. Show why the operator enforces the intended condition, including mixed and boundary cases where they matter. For example, explain why taking the maximum of two gains keeps the scale large when only one error is small. Make clear whether a transition is gradual or binary, whether a threshold includes equality, and what a zero update preserves when these affect behavior.
- For mechanisms with memory, show what persists, how it is initialized and updated, and when it resets. Explain behavior across time, not only at one instant. For a multi-term reward or objective, distinguish dependencies on current state, change since the last step, or the best result so far, as applicable. Trace a short sequence that includes progress, loss, recovery, and holding when relevant; show which terms apply together and whether displayed values are individual terms or the total. Do not imply that defining each term separately explains their interaction.
- Explain how neighboring components divide the work. State which part of the pipeline each changes and which remaining problem motivates the next component. For example, making a small error noticeable in the input and reducing the size of an output correction solve different problems. Similarly, attaining a new best, recovering an old best, and maintaining acceptable performance require different explanations.
- Use tables, diagrams, and summaries together with prose that explains the mechanism. A component catalog, reward table, or architecture figure must not leave the reader to reconstruct the reasoning. Keep explanations concise by removing repetition and incidental detail, not the intermediate reasoning needed to follow them.

#### Evidence and Presentation

- Cover the aspects requested for the specific paper, such as policy structure, datasets, training settings, inference, ablations, and evaluation results. Clearly distinguish pre-training, post-training, fine-tuning, and zero-shot evaluation where applicable.
- Explain what each important experiment tests, what the comparison controls, what its metric means, and how the result supports or limits the paper's claim. Interpret selected numbers instead of presenting an unexplained catalog of tables. Distinguish reported findings from explanatory examples or your own interpretation.
- Label invented numerical examples as illustrative, state simplifications that affect their interpretation, and check their arithmetic against the actual method. Do not silently apply simulation settings to hardware, present a continuous rule as a binary switch, or turn a possible effect into a guaranteed result. Preserve technical accuracy while simplifying the explanation; avoid extensive copying from the paper.
- Add only figures that materially improve understanding. Crop them from the paper, store them under `assets/<Note_Name>/`, reference them with repository-relative paths, and add a short descriptive caption. Explain the necessary content in the note; source figure, equation, and section numbers are attribution, not substitutes for an explanation.
- Update the relevant `README.md` index and file tree when adding, renaming, moving, or removing a paper note.

#### Review for Understanding Before Finishing

- Read the note continuously as someone who has not read the paper. For every central mechanism, check that the reader can explain the failure it addresses, follow a concrete input through its steps, predict the outcome, and explain why the simpler alternative is inadequate. Revise any answer that still depends on reasoning available only in the PDF or in the writer's head.
- Check whether the reader can distinguish related components, follow persistent state across updates, and explain an important mixed, zero, or boundary case where applicable. Verify that the examples agree with the equations and that the experimental evidence supports the stated conclusions. Defined terminology and successful Markdown rendering alone do not establish that the explanation is understandable.
- When a reader asks for clarification, identify the missing reasoning and review related passages within the requested scope for the same problem. Repair the explanation at that level instead of only adding a definition or rephrasing the confusing sentence. Apply the lesson when drafting subsequent notes; the reader should not have to supply the explanatory structure or worked examples.
- Check that equations, figures, tables, and links render correctly after the content review.

### Book Notes

- Base each note directly on the book PDF supplied by the user.
- Build book notes incrementally, one requested chapter at a time. Append new chapters to the existing book note rather than creating one file per chapter.
- Keep a navigable catalog or table of contents at the top of each book note and update it whenever a chapter is added.
- Do not extensively copy the chapter. Explain its core material precisely, prioritizing important definitions, formulas, algorithms, and short clarifying examples.
- Make each chapter note self-contained for review without reopening the book PDF. Include the definitions, symbol meanings, assumptions, and intermediate reasoning needed to understand the material; link to earlier sections of the note for prerequisites already covered.
- Do not rely on a figure, equation, example, or appendix that appears only in the source book. Embed necessary figures near the relevant explanation, or provide an adequate explanation in the note. Keep source numbers and page references as attribution, not as substitutes for the content.
- Add figures from the book only when they are necessary for a clear explanation. Store extracted figures under the corresponding book asset directory and provide descriptive captions.
- Maintain consistent notation across chapters and with the source book. Check that equations, Mermaid diagrams, code fences, anchors, and relative links render correctly.
- Current book-note locations are:
  - robotics: `docs/Foundations/Books/Robotics/Modern_Robotics.md`
  - reinforcement learning: `docs/Foundations/Books/Reinforcement_Learning/Reinforcement_Learning_An_Introduction.md`

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
