from transformers import pipeline
import gradio as gr

model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

def sent(text):
    return sentiment_task(text)[0]['label'], sentiment_task(text)[0]['score']


inputs = gr.inputs.Textbox(lines=5, label="Input Text")

outputs = [
           gr.outputs.Label(label="label"),
           gr.outputs.Label(label="Score")
]


title = "XLM-T"
description = "demo for Cardiff NLP XLM-T. To use it, simply add your text, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2104.12250'>XLM-T: A Multilingual Language Model Toolkit for Twitter</a> | <a href='https://github.com/cardiffnlp/xlm-t'>Github Repo</a></p>"
examples = [
    ["Huggingface es lo mejor! Awesome library 🤗😎"]
    ["نوال الزغبي (الشاب خالد ليس عالمي) هههههههه أتفرجي على ها الفيديو يا مبتدئة http vía @user"],
    ["Royal: le président n'aime pas les pauvres? c'est n'importe quoi http …"],
    ["@user korrekt! Verstehe sowas nicht..."]
]

gr.Interface(sent, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()