## Choptimize

[![License][license-img]][license-url]&nbsp;
[![Python][py-img]][py-url]

A multimodal tool for analysing and optimizing coding prompts, grounded in the [prompt2code-eval](https://huggingface.co/datasets/darragh0/prompt2code-eval) dataset.

> [!IMPORTANT]
> Requires Python 3.13+

## Background

This tool was created as a research artefact from an investigation into the correlation between prompt quality and code quality in developer–LLM code conversations, where we created and analysed the [prompt2code-eval](https://huggingface.co/datasets/darragh0/prompt2code-eval) dataset (~26K scored prompt–code pairs).

The source code for the tool is in `app/`; the preprocessing pipeline that produced the dataset is in `preproc/`.

### Structure

| Package      | Description                              |
| ------------ | ---------------------------------------- |
| `preproc/`   | Dataset creation: preprocessing pipeline |
| `analysis/`  | Statistical analysis                     |
| `app/`       | CLI / web app tool                       |

### Preprocessing

| Script                | Description                                |
| --------------------- | ------------------------------------------ |
| `preproc/download.py` | Download CodeChat-V2.0 dataset             |
| `preproc/filter.py`   | Filter to English prompts with Python code |
| `preproc/syntax.py`   | Syntactic analysis (ruff & radon)          |
| `preproc/semantics.py`| Semantic analysis of prompt-code pairs     |

See: [`preproc/README.md`](./preproc/README.md)

!!! TODO: run instructions

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
