# Notices and Attributions

## Code
The source code in this repository is licensed under MIT — see `LICENSE`.

## Model: e5-base-v2
Retrieval currently runs on [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2),
released under the MIT License by its authors, used unmodified (no fine-tuning).
Weights are downloaded directly from Hugging Face at setup time and are not
redistributed in this repository. See `eval/` for retrieval-quality measurements
against the previous embedding approach.

> Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., &
> Wei, F. (2022). Text Embeddings by Weakly-Supervised Contrastive
> Pre-training. *arXiv:2212.03533*.

### Previously evaluated: InLegalBERT
Earlier versions of this project used [law-ai/InLegalBERT](https://huggingface.co/law-ai/InLegalBERT)
(MIT License), mean-pooled with no fine-tuning applied. The eval harness showed
this performed substantially worse than e5-base-v2 on this dataset — it is no
longer part of the live retrieval pipeline. If referencing that approach, cite:

> Paul, S., Mandal, A., Goyal, P., & Ghosh, S. (2023). Pre-trained Language
> Models for the Legal Domain: A Case Study on Indian Law. *Proceedings of the
> 19th International Conference on Artificial Intelligence and Law (ICAIL
> 2023)*.

## Data
The dataset (`Data/finalcases.json`, `Data/QandA.jsonl`) is original written
analysis and is **not published in this repository**. It is not covered by
the MIT license above.

## Disclaimer
LegalisAI is an informational and educational tool. It does not provide legal
advice, and its output should not be relied upon as a substitute for
consultation with a qualified legal professional. Retrieval results and
similarity scores reflect a statistical model and may be incomplete, outdated,
or inapplicable to any specific situation.
