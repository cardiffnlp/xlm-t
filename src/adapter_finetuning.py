import os
import shutil
from glob import glob
from pathlib import Path
import datetime
from collections import defaultdict
import urllib

import numpy as np
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer
from transformers import AdapterType
from transformers import AutoConfig, AutoModelWithHeads
from transformers import TrainingArguments, Trainer, EvalPrediction

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

import argparse
from pathlib import Path
import datetime
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

# --- PARAMS ---

parser = argparse.ArgumentParser(description='List the content of a folder')

parser.add_argument('--language', default="spanish", type=str, help='languages: arabic, english, ..., all')
parser.add_argument('--model', default="cardiffnlp/twitter-xlm-roberta-base", type=str, help='hugging face model or path to local model')
parser.add_argument('--seed', default=1, type=int, help='...')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--max_epochs', default=20, type=int, help='Number of training epochs (will train all of them then select the best one)')
# There is no early stopping in the transformers version of adaptors so we need to set
# a maximum number of epochs (and select the best model when training finishes)

args = parser.parse_args()

LANGUAGE = args.language
MODEL= args.model
SEED= args.seed
LR = args.lr
MAX_EPOCHS = args.max_epochs

# Fixed params
EVAL_STEPS = 20
BATCH_SIZE = 200
NUM_LABELS = 3

now = datetime.datetime.now()
now = now.strftime('%Y%m%d_%H%M%S_%f')

UNIQUE_NAME = f"{LANGUAGE}_{MODEL.replace('//','-')}_{LR}_{SEED}_{now}"
UNIQUE_NAME = UNIQUE_NAME.replace('.','-')
DIR = f"./{UNIQUE_NAME}/"
Path(DIR).mkdir(parents=True, exist_ok=True)


# --- LOAD DATA ---
    
language = 'spanish'

files = """test_labels.txt
test_text.txt
train_labels.txt
train_text.txt
val_labels.txt
val_text.txt""".split('\n')

def fetch_data(language, files):
 dataset = defaultdict(list)
 for infile in files:
   thisdata = infile.split('/')[-1].replace('.txt','')
   dataset_url = f"https://raw.githubusercontent.com/cardiffnlp/xlm-t/main/data/sentiment/{language}/{infile}"
   print(f'Fetching from {dataset_url}')
   with urllib.request.urlopen(dataset_url) as f:
     for line in f:
       if thisdata.endswith('labels'):
         dataset[thisdata].append(int(line.strip().decode('utf-8')))
       else:
         dataset[thisdata].append(line.strip().decode('utf-8'))
 return dataset     

dataset_dict = fetch_data(language, files)

dataset = DatasetDict()

for split in ['train', 'val', 'test']:
    d = {"text":dataset_dict[f'{split}_text'], 'labels':dataset_dict[f'{split}_labels']}
    if split == 'val':
        split = 'validation' #name mismatch with xlm-t dataset and library datasets
    dataset[split] = Dataset.from_dict(d)
    

# --- MODEL ---

config = AutoConfig.from_pretrained(
    MODEL,
    num_labels=NUM_LABELS,
)
model = AutoModelWithHeads.from_pretrained(
    MODEL,
    config=config,
)

# Add a new adapter
adapter_name = f"adapter_{UNIQUE_NAME}" 
#adapter_name = f"xlm-t-sentiment"
model.add_adapter(adapter_name, AdapterType.text_task)

# Add a matching classification head
model.add_classification_head(
    adapter_name,
    num_labels=NUM_LABELS,
    id2label={ 0: "Neg", 1:"Neu", 2:"Pos"}
  )

# Activate the adapter
model.train_adapter(adapter_name)


# --- TRAINING ---

training_args = TrainingArguments(
    learning_rate=LR,
    num_train_epochs=MAX_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_steps=10,
    output_dir=DIR,
    overwrite_output_dir=True,
    remove_unused_columns=False,
    seed=SEED,
    load_best_model_at_end=True,
    do_eval=True,
    eval_steps=EVAL_STEPS,
    evaluation_strategy="steps"
)

global val_history
val_history = []
def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    f1 = f1_score(p.label_ids, preds, average='macro')
    acc = (preds == p.label_ids).mean()
    val_history.append(f1)
    return {"macro_f1":f1, "acc": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_accuracy,
)

trainer.train()
trainer.evaluate()
best_step = val_history.index(val_history.pop())+1
n_steps = len(val_history)

print(val_history)
print(best_step)

model.save_adapter(DIR+adapter_name, adapter_name)

# remove checkpoints
checkpoints = glob(DIR+"/check*")
for c in checkpoints:
    print('Removing:',c)
    shutil.rmtree(c)

    
# --- EVALUATION ---

#test_labels = dataset["test"]["labels"]
test_preds_raw, test_labels , out = trainer.predict(dataset["test"])
test_preds = np.argmax(test_preds_raw, axis=-1)

print(out)
test_preds_raw_path = f"{DIR}preds_test.txt"
np.savetxt(test_preds_raw_path, test_preds_raw)

print(classification_report(test_labels, test_preds, digits=3))