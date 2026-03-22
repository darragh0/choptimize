# choptimize

CLI tool for analysing prompt quality in developer-LLM code conversations, using the [CodeChat-V2.0](https://huggingface.co/datasets/Suzhen/CodeChat-V2.0) dataset (587k conversations from [WildChat](https://huggingface.co/datasets/allenai/WildChat)).

> [!IMPORTANT]
> Requires Python 3.13+

### Structure

| Package      | Description                                        |
| ------------ | -------------------------------------------------- |
| `preproc/`   | Data preprocessing pipeline (download, filter, evaluate) |
| `analysis/`  | Statistical analysis (correlations, visualizations) |

### Preprocessing

| Script                | Description                                |
| --------------------- | ------------------------------------------ |
| `preproc/download.py` | Download the CodeChat-V2.0 dataset         |
| `preproc/filter.py`   | Filter to English prompts with Python code |
| `preproc/syntax.py`   | Syntactic analysis (ruff & radon)          |
| `preproc/semantics.py`| Semantic analysis of prompt-code pairs     |

### Analysis

```sh
uv run python -m analysis
```

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
