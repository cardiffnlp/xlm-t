This repository contains data, code and pre-trained multilingual language models for Twitter.

# XLM-R-Twitter

As explained in the reference paper, we make use of XLM-R-base () and continue pre-training on a large corpus of Twitter in multiple languages. This masked language model (XLM-R-Twitter) can be downloaded from 🤗HuggingFace [here](XXX). 

**Extra**: A similar language model but based on RoBERTa and English-language Twitter data is available [here](https://huggingface.co/cardiffnlp/twitter-roberta-base). 

We also provide task-specific models:

TOCOMPLETE XXX

## Code

We include code with various functionalities around our released language models. TODO: Add code or links to code for finetuning, analysis, embeddings, etc.

# Cross-lingual Sentiment Analysis: The Benchmark

As part of our framework, we release a unified benchmark for cross-lingual sentiment analysis for XXX different languages. All datasets are framed as tweet classification with three labels (positive, negative and neutral). The languages available are XXX (TODO: Add languages/citations). The format for each dataset follows that of [TweetEval](https://github.com/cardiffnlp/tweeteval) with one line per tweet and label. 

# Reference paper

If you use this repository in your research, please use the following `bib` entry to cite the reference paper.

```
@inproceedings{barbieri2021xlmtwitter,
  title={{Multilingual Language Models in Twitter}},
  author={Barbieri, Francesco and  Espinosa-Anke, Luis and Camacho-Collados, Jose},
  booktitle={Submitted to ACL Demo},
  year={2021}
}
```
# License

This repository is released open-source but but restrictions may apply to individual datasets (which are derived from existing data) or Twitter (main data source). We refer users to the original licenses accompanying each dataset and Twitter regulations.
