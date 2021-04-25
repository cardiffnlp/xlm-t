# usage: evaluaton_script.py [-h] [--gold_path gold_path]
#                            [--predictions_path PREDICTIONS_PATH] [--language LANGUAGE]

# optional arguments:
#   -h, --help: show this help message and exit
#   --gold_path: Path to umsab dataset
#   --predictions_path: Path to predictions file
#   --language: language to evaluate ('arabic', 'english', ... or 'all')

import argparse
import os
from sklearn.metrics import classification_report
import numpy as np

all_languages = ['arabic', 'english', 'french', 'german', 'hindi', 'italian', 'spanish', 'portuguese']

def load_gold_pred(args):
    gold_path = args.gold_path
    predictions_path = args.predictions_path
    language = args.language

    gold_path = os.path.join(gold_path,language,'test_labels.txt')
    pred_path = os.path.join(predictions_path,language+'.txt')
    gold = open(gold_path).read().split("\n")[:-1]
    pred = open(pred_path).read().split("\n")[:-1]
        
    return gold, pred

def single_language_result(args):
    gold, pred = load_gold_pred(args)
    results = classification_report(gold, pred, output_dict=True)
    return results['macro avg']['f1-score']

def all_languages_results(args):
    all_results = {}
    gold, pred = load_gold_pred(args)

    # in all/test_labels.txt the languages are in order: 
    # arabic: line 0 to 868, english 869 to 1738, ...
    single_lang_test_size = 869 #size of each language test set
    a, b = 0, 0
    for i in range(len(all_languages)):  
        a = (i)*single_lang_test_size
        b = (i+1)*single_lang_test_size  
        results = classification_report(gold[a:b], pred[a:b], output_dict=True)
        r = results['macro avg']['f1-score']
        all_results[all_languages[i]] = r

    return all_results


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='umsab evaluation script.')
    
    parser.add_argument('--gold_path', default="./data/sentiment/", type=str, help='Path to umsab dataset')
    parser.add_argument('--predictions_path', default="./predictions/sentiment/", type=str, help='Path to predictions files')
    parser.add_argument('--language', default="all", type=str, help="Language to evaluate ('arabic', 'english', ... or 'all')")

    args = parser.parse_args()

    language = args.language
    if language == 'all':
        all_results = all_languages_results(args)
        for k in all_results:
            print(f"{k}: {all_results[k]}")
        #print(f"Avg: {list(all_results.values())}")
        print(f"Avg: {np.mean(list(all_results.values()))}")

    else:
        result = single_language_result(args)
        print(f"{language}: {result}")


