This is the **XLM-T** repository, wich data, code and pre-trained multilingual language models for Twitter.

# XLM-T - A Multilingual Language Model Toolkit for Twitter

As explained in the reference paper, we make start from [XLM-Roberta base](https://huggingface.co/transformers/model_doc/xlmroberta.html) and continue pre-training on a large corpus of Twitter in multiple languages. This masked language model, which we have named `twitter-xlm-roberta-base` in the 🤗Huggingface hub, can be downloaded from [here](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base). 

**Note**: This Twitter-specific pretrained LM was pretrained following a similar strategy to its English-only counterpart, which was introduced as part of the [TweetEval](https://github.com/cardiffnlp/tweeteval) framework, and available [here](https://huggingface.co/cardiffnlp/twitter-roberta-base).

We also provide task-specific models based on the [Adapter](https://adapterhub.ml/) technique, fine-tuned for **cross-lingual sentiment analysis** (See #3):

# 1 - Code

We include code with various functionalities to complement this release. We provide examples for, among others, feature extraction and adapter-based inference with language models in this [notebook](https://github.com/cardiffnlp/xlm-t/blob/main/notebooks/twitter-xlm-roberta-base.ipynb). Also with examples for training and evaluating language models on multiple tweet classification tasks, compatible with `UMSAB` (see `#2`) and [TweetEval](https://github.com/cardiffnlp/tweeteval/tree/main/datasets) datasets.

## Perform inference with Huggingface's _pipelines_

Using Huggingface's `pipelines`, obtaining predictions is as easy as:

```python
from transformers import pipeline
model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
sentiment_task("Huggingface es lo mejor! Awesome library 🤗😎")
```
```
[{'label': 'Positive', 'score': 0.9343640804290771}]
```

## Fine-tune `xlm-t` with _adapters_

You can fine-tune an adapter built on top of your language model of choice by running the `src/adapter_finetuning.py` script, for example:

```
python3 src/adapter_finetuning.py --language spanish --model cardfiffnlp/twitter-xlm-roberta-base --seed 1 --lr 0.0001 --max_epochs 20
```

## Colab notebook

For quick prototyping, you can direclty use the Colab notebook provided [here](https://colab.research.google.com/drive/1pGUCW250eHbzIQiENdVx2n65ZJADOi80?usp=sharing). For more details on training and evaluating models on TweetEval tweet classification tasks, [this notebook](https://github.com/cardiffnlp/tweeteval/blob/main/TweetEval_Tutorial.ipynb) provides additional support.

# 2 - `UMSAB`, the Unified Multilingual Sentiment Analysis Benchmark

As part of our framework, we also release a unified benchmark for cross-lingual sentiment analysis for eight different languages. All datasets are framed as tweet classification with three labels (positive, negative and neutral). The languages included in the benchmark, as well as the datasets they are based on, are: Arabic (SemEval-2017, [Rosenthal et al. 2017](https://www.aclweb.org/anthology/S17-2088.pdf)), English (SemEval-17, [Rosenthal et al. 2017](https://www.aclweb.org/anthology/S17-2088.pdf)), French (Deft-2017, [Benamara et al. 2017](https://oatao.univ-toulouse.fr/19108/1/benamara_19108.pdf)), German (SB-10K, [Cieliebak et al. 2017](https://www.aclweb.org/anthology/W17-1106.pdf)), Hindi (SAIL 2015, [Patra et al. 2015](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.728.5241&rep=rep1&type=pdf)), Italian (Sentipolc-2016, [Barbieri et al. 2016](https://hal.inria.fr/hal-01414731/file/paper_026.pdf)), Portuguese (SentiBR, [Brum and Nunes, 2017](https://www.aclweb.org/anthology/L18-1658.pdf)) and Spanish (Intertass 2017, [Díaz Galiano et al. 2018](https://rua.ua.es/dspace/bitstream/10045/74613/1/PLN_60_04.pdf)). The format for each dataset follows that of *TweetEval* with one line per tweet and label per line. 

## `UMSAB` Results / Leaderboard

The following results (Macro F1 reported) correspond to XLM-R (Conneau et al. 2020) and XLM-Tw, the same model retrained on Twitter as explained in the reference paper. The two settings are monolingual (trained and tested in the same language) and multilingual (considering all languages for training).  Check the reference paper for more details on the setting and the metrics.

|     | FT Mono | XLM-R Mono | XLM-Tw Mono | XLM-R Multi | XLM-Tw Multi |
|-----|---------|-------------|-------------|--------------|--------------|
| **Arabic**  |   46.0  |     63.6    |     67.7    |     64.3     |     66.9     |
| **English**  |   50.9  |     68.2    |     66.9    |     68.5     |     70.6     |
| **French**  |   54.8  |     72.0    |     68.2    |     70.5     |     71.2     |
| **German**  |   59.6  |     73.6    |     76.1    |     72.8     |     77.3     |
| **Hindi**  |   37.1  |     36.6    |     40.3    |     53.4     |     56.4     |
| **Italian**  |   54.7  |     71.5    |     70.9    |     68.6     |     69.1     |
| **Portuguese**  |   55.1  |     67.1    |     76.0    |     69.8     |     75.4     |
| **Spanish**  |   50.1  |     65.9    |     68.5    |     66.0     |     67.9     |
| *All lang.* |   *51.0*  |     *64.8*    |     *66.8*    |     *66.8*     |     *69.4*     |

If you would like to have your results added to the leaderboard you can either submit a pull request or send an email to any of the paper authors with results and the predictions of your model. Please also submit a reference to a paper describing your approach.

## Evaluating your system

For evaluating your system, you simply need an individual prediction file for each of the languages. The format of the predictions file should be the same as the output examples in the predictions folder (one output label per line as per the original test file) and the files should be named *language.txt* (e.g. *arabic.txt*). The predictions included as an example in this repo correspond to the `all` dataset (*All lang.*).

## Example usage

```
python src/evaluation_script.py
```

The script takes as input a set of test labels and the predictions from the "predictions" folder by default, but you can set this to suit your needs as optional arguments.

## Optional arguments

Three optional arguments can be modified:

*--gold_path*: Path to gold datasets. Default: `./data/sentiment`

*--predictions_path*: Path to predictions directory. Default: `./predictions/sentiment`

*--language*: Language to evaluate (`arabic`, `english` ... or `all`). Default: `all`

Evaluation script sample usage from the terminal with parameters:

```bash
python src/evaluation_script.py --language arabic
```
(this script would output the results for the Arabic dataset only)

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

If using `UMSAB`, please also cite their corresponding datasets.

# License

This repository is released open-source but but restrictions may apply to individual datasets (which are derived from existing data) or Twitter (main data source). We refer users to the original licenses accompanying each dataset and Twitter regulations.
