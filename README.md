## Choptimize

[![License][license-img]][license-url]&nbsp;
[![Python][py-img]][py-url]

A tool for analysing and optimizing coding prompts, grounded in the [prompt2code-eval](https://huggingface.co/datasets/darragh0/prompt2code-eval) dataset (~26K scored prompt–code pairs).

Built as a research artefact investigating the correlation between prompt quality and code quality in developer–LLM conversations.

### Install

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/darragh0/choptimize.git
cd choptimize
uv sync
```

### Usage

**CLI**

```
uv run python -m app "Your prompt here"
```

Options:

- `-i, --improve` — generate an improved version of the prompt
- `-s, --service {ollama,openai,gemini}` — LLM service (default: ollama)
- `-m, --model NAME` — model name
- `-k, --api-key KEY` — API key (required for openai/gemini)

Example:

```
uv run python -m app "Write a Python function that sorts a list" -s ollama -i
```

**Web**

```
uv run python -m app web
```

Opens at `http://127.0.0.1:8000`.

### Project structure

| Directory    | Description                              |
| ------------ | ---------------------------------------- |
| `app/`       | CLI and web interface                    |
| `preproc/`   | Dataset preprocessing pipeline           |
| `analysis/`  | Statistical analysis and visualisations  |
| `eval/`      | Evaluation suite                         |

### Preprocessing

The `preproc/` pipeline builds the prompt2code-eval dataset from CodeChat-V2.0:

| Script                 | Description                                |
| ---------------------- | ------------------------------------------ |
| `preproc/download.py`  | Download CodeChat-V2.0 dataset             |
| `preproc/filter.py`    | Filter to English prompts with Python code |
| `preproc/syntax.py`    | Syntactic analysis (ruff & radon)          |
| `preproc/semantics.py` | Semantic analysis of prompt–code pairs     |

See [`preproc/README.md`](./preproc/README.md) for details.

### Citation

```
@misc{zhong2025developerllmconversationsempiricalstudy,
      title={Developer-LLM Conversations: An Empirical Study of Interactions and Generated Code Quality},
      author={Suzhen Zhong and Ying Zou and Bram Adams},
      year={2025},
      eprint={2509.10402},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2509.10402},
}
```

### License

[MIT](https://github.com/darragh0/choptimize?tab=MIT-1-ov-file)

[license-img]: https://img.shields.io/github/license/darragh0/choptimize?style=flat-square&logo=apache&label=%20&color=red
[license-url]: https://github.com/darragh0/choptimize?tab=MIT-1-ov-file#
[py-img]: https://img.shields.io/badge/3.13%2B-blue?style=flat-square&logo=python&logoColor=FFFD85
[py-url]: https://www.python.org/
