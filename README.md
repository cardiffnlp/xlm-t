This is the **XLM-T** repository, wich data, code and pre-trained multilingual language models for Twitter.

# XLM-T - A Multilingual Language Model Toolkit for Twitter

As explained in the reference paper, we make start from [XLM-Roberta base](https://huggingface.co/transformers/model_doc/xlmroberta.html) and continue pre-training on a large corpus of Twitter in multiple languages. This masked language model, which we have named `twitter-xlm-roberta-base` in the 🤗Huggingface hub, can be downloaded from [here](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base). 

**Note**: This Twitter-specific pretrained LM was pretrained following a similar strategy to its English-only counterpart, which was introduced as part of the [TweetEval](https://github.com/cardiffnlp/tweeteval) framework, and available [here](https://huggingface.co/cardiffnlp/twitter-roberta-base).

We also provide task-specific models based on the [Adapter](https://adapterhub.ml/) technique, fine-tuned for **cross-lingual sentiment analysis** (See #3):

# 1 - Code

We include code with various functionalities to complement this release. Minimal start examples for feature extraction and adapter-based inference are available in this [notebook](https://github.com/cardiffnlp/xlm-t/blob/main/notebooks/twitter-xlm-roberta-base.ipynb). 

```python
from transformers import pipeline
model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
sentiment_task("Huggingface es lo mejor! Awesome library 🤗😎")
>>> [{'label': 'Positive', 'score': 0.9343640804290771}]
```

# 2 - Cross-lingual Sentiment Analysis: The Benchmark

As part of our framework, we also release a unified benchmark for cross-lingual sentiment analysis for eight different languages. All datasets are framed as tweet classification with three labels (positive, negative and neutral). The languages available we include are: Arabic, English, French, German, Hindi, Italian, Portuguese and Spanish. The format for each dataset follows that of *TweetEval* with one line per tweet and label per line. 

# Reference paper

If you use this repository in your research, please use the following `bib` entry to cite the reference paper.

```
@inproceedings{barbieri2021xlmtwitter,
  title={{A Multilingual Language Model Toolkit for Twitter}},
  author={Barbieri, Francesco and  Espinosa-Anke, Luis and Camacho-Collados, Jose},
  booktitle={Submitted to ACL Demo},
  year={2021}
}
```
# License

This repository is released open-source but but restrictions may apply to individual datasets (which are derived from existing data) or Twitter (main data source). We refer users to the original licenses accompanying each dataset and Twitter regulations.
