# Choptimize Progress

## What Is This?

FYP (CS4437, UL 2025/26): UL-style research paper (10k words max) + software deliverable.
**RQ:** Does prompt quality correlate with code quality in LLM-generated code?
**Tool:** Choptimize — CLI/web app that scores developer prompts using RAG over a scored dataset + local LLM.

## Key Info

|                |                                                                                                   |
| -------------- | ------------------------------------------------------------------------------------------------- |
| Student        | Darragh L. (23359048)                                                                             |
| Supervisor     | Dr. Salaheddin A. (Wed 4pm)                                                                       |
| Deadline       | **April 7, 2026**                                                                                 |
| Dataset        | [darragh0/prompt2code-eval](https://huggingface.co/datasets/darragh0/prompt2code-eval) — 27K rows |
| Prompt dims    | clarity, specificity, completeness (1–5)                                                          |
| Code dims      | correctness, robustness, readability, efficiency (1–5)                                            |
| Syntax metrics | ruff (errors, warnings, flake8, bugbear, security), radon (complexity, maintainability)           |

## Where Things Live

| Location                                          | What                                                  |
| ------------------------------------------------- | ----------------------------------------------------- |
| `choptimize/`                                     | All code: preprocessing, analysis, tool               |
| `common/src/common/utils/`                        | Shared utilities (cache, dataset loading, console)    |
| `preproc/`                                        | Data pipeline (download → filter → syntax → semantic) |
| `analysis/`                                       | Dataset analysis (correlations + visualizations)      |
| `data/`                                           | Git-ignored cache directory                           |
| `~/fyp/final-report.current-state.unfinished.txt` | Current report draft                                  |

**⚠️ Ignore:** interim report — outdated.

## Report Progress

**Word budget:** 10,000 max (references included). ~7,000 used. ~3,000 remaining.

| #   | Section                           | Weight   | Words | Status                                     |
| --- | --------------------------------- | -------- | ----- | ------------------------------------------ |
| —   | Abstract                          | —        | 234   | ✅ Done                                    |
| 1   | Introduction                      | 10%      | 605   | ✅ Done                                    |
| 2   | Background & Literature Review    | 20%      | 1,733 | ✅ Done                                    |
| 3   | Methodology                       | 10%      | 738   | ✅ Done                                    |
| 4   | Implementation & Primary Research | **25%**  | 1,567 | ✅ Done                                    |
| 5   | Results & Analysis                | 10%      | 1,228 | ✅ Done                                    |
| 6   | Discussion & Conclusions          | 10%      | 899   | ✅ Done                                    |
| —   | References                        | (in 15%) | —     | ⬜ Stored elsewhere — will be pasted later |

---

## Baby Steps — What To Do Next

Do these **one at a time, in order**. Don't look ahead. Just do the current one.

### ✅ Dataset Analysis — Complete

### ✅ Build Choptimize Tool — Complete (CLI + Web)

### 🔵 Final Steps

- [x] **Step 1:** Add scoring rubric table + antipattern catalogue table to report
- [x] **Step 2:** Add CLI/web screenshots to report (weak prompt, strong prompt, web UI in appendix)
- [ ] **Step 3:** Add runtime architecture diagram
- [x] **Step 4:** Paste references
- [ ] **Step 5:** Final pass: word count check, cross-references, reference formatting
- [ ] **Step 6:** Record video summary

---

_Updated by Claude Code when you confirm a step is complete._
