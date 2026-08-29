# Notices and Attributions

## Code
The source code in this repository is licensed under MIT — see `LICENSE`.

## Model: InLegalBERT
This project uses [law-ai/InLegalBERT](https://huggingface.co/law-ai/InLegalBERT),
released under the MIT License by its authors. Base model weights are
downloaded directly from Hugging Face at setup time and are not redistributed
in this repository. The fine-tuned checkpoints in `legalis_model/` and
`faq_model/` (gitignored, not published) are derived from InLegalBERT.

If you use this project or its approach, please also cite the original work:

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
