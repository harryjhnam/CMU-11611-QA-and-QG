#!/usr/bin/env python3

import os
import json
from glob import glob
import pandas as pd
from tqdm import tqdm

from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu

import argparse

from QA import get_window_paragraph, boolean_classify, ext_QA, gen_QA


def evaluate(qa_dataset, articles):

    results = {
        'document_ids': [],
        'questions': [],
        'answers': [],
        'ours_preds': [],
        'roberta_preds': [],
        't5_preds': [],
        'roberta_window_preds': [],
        't5_window_preds': []
    }

    for document_id, qa_pairs in tqdm(qa_dataset.items()):
        document = articles[document_id]

        for question, answer in tqdm(qa_pairs, leave=False):
        
            # Append document id, question, answer
            results['document_ids'].append(document_id)
            results['questions'].append(question)
            results['answers'].append(answer.lower())

            # roberta baseline
            results['roberta_preds'].append(ext_QA(question, document))

            # t5 baseline
            results['t5_preds'].append(gen_QA(question, document))

            # ours and window baselines
            window_paragraph = get_window_paragraph(document, question)

            roberta_window_pred = ext_QA(question, window_paragraph)
            results['roberta_window_preds'].append(roberta_window_pred)
            t5_window_pred = gen_QA(question, window_paragraph)
            results['t5_window_preds'].append(t5_window_pred)

            if boolean_classify(question) == "LABEL_0":
                results['ours_preds'].append(roberta_window_pred)
            else:
                results['ours_preds'].append(t5_window_pred)
                
    n_samples = len(results['document_ids'])
    for k, l in results.items():
        assert n_samples == len(l)

    return results


def calculate_bleu_score(results):
    
    bleu_sum = defaultdict(lambda: 0.)

    for answer, ours, roberta, t5, w_roberta, w_t5\
        in tqdm(zip(results['answers'], results['ours_preds'], results['roberta_preds'], results['t5_preds'], results['roberta_window_preds'], results['t5_window_preds'])):

        answer = [answer.lower().split(' ')]

        bleu_sum['ours_preds'] += sentence_bleu( answer, ours.lower().split(' '), weights=(1,0,0,0) )
        bleu_sum['roberta_preds'] += sentence_bleu( answer, roberta.lower().split(' '), weights=(1,0,0,0) )
        bleu_sum['t5_preds'] += sentence_bleu( answer, t5.lower().split(' '), weights=(1,0,0,0) )
        bleu_sum['roberta_window_preds'] += sentence_bleu( answer, w_roberta.lower().split(' '), weights=(1,0,0,0) )
        bleu_sum['t5_window_preds'] += sentence_bleu( answer, w_t5.lower().split(' '), weights=(1,0,0,0) )

    n_samples = len(results['document_ids'])
    bleu_scores = {k: v/n_samples for k, v in bleu_sum.items()}

    return bleu_scores


def main(args):
    # Load articles
    articles_paths = glob(os.path.join(args.articles_dir, "*/*.txt"))
    articles_ids = ["/"+os.path.join( os.path.basename(os.path.dirname(path)), os.path.basename(path) )
                for path in articles_paths]
    articles = {id: open(path, "r").read() for path, id in zip(articles_paths, articles_ids)}

    # Load QA dataset
    qa_dataset = pd.read_csv(args.qa_dataset_path, header=0, sep=",", index_col=0)
    e_dataset = qa_dataset[qa_dataset["difficulty"] == "E"]
    m_dataset = qa_dataset[qa_dataset["difficulty"] == "M"]
    h_dataset = qa_dataset[qa_dataset["difficulty"] == "H"]

    # Group questions by difficulty
    # each is a dictionary with key = document_id and value = list of tuples (question, answer)
    easy_qa = e_dataset[['question', 'answer']].agg(tuple, 1).groupby(e_dataset['document_id']).apply(list).to_dict()
    medium_qa = m_dataset[['question', 'answer']].agg(tuple, 1).groupby(m_dataset['document_id']).apply(list).to_dict()
    hard_qa = h_dataset[['question', 'answer']].agg(tuple, 1).groupby(h_dataset['document_id']).apply(list).to_dict()

    # Evaluate on each difficulty
    all_results = defaultdict(list)

    bleu_scores = {}

    for difficulty, qa_dataset in [('E', easy_qa), ('M', medium_qa), ('H', hard_qa)]:
        result_json_path = os.path.join(args.evaluation_output_dir, f'{difficulty}_predictions.json')

        if os.path.exists(result_json_path):
            print(f"--> Loading inference results from file {result_json_path}")
            with open(result_json_path, 'r') as f:
                results = json.load(f)
        else:
            print(f"--> Inferencing on {difficulty} questions")
            results = evaluate(qa_dataset, articles)
            with open(result_json_path, 'w') as f:
                json.dump(results, f, indent=4)
        print()

        # Calculate bleu score
        print(f"--> Calculating BLEU score on {difficulty} questions")
        bleu_scores[difficulty] = calculate_bleu_score(results)
        print("    :: BLEU scores:")
        for k, v in bleu_scores[difficulty].items():
            print(f"    :: - {k}: {v}")
        print()

        # Add to all results
        for k, l in results.items():
            all_results[k].extend(l)

    # Calculate bleu score for all results
    print(f"--> Calculating BLEU score on all questions")
    bleu_scores['all'] = calculate_bleu_score(all_results)
    print("    :: BLEU scores:")
    for k, v in bleu_scores['all'].items():
        print(f"    :: - {k}: {v}")

    # Save bleu scores
    bleu_scores_path = os.path.join(args.evaluation_output_dir, 'bleu_scores.json')
    with open(bleu_scores_path, 'w') as f:
        json.dump(bleu_scores, f, indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=''' Question Answering & Generation Pipeline''')
    
    parser.add_argument('--articles_dir', type=str, default = "data/documents/")
    parser.add_argument('--qa_dataset_path', type=str, default = "data/qa_dataset-0.1.csv")
    
    parser.add_argument('--evaluation_output_dir', type=str, default = "data/eval_results/")
    
    args = parser.parse_args()

    if not os.path.exists(args.evaluation_output_dir):
        os.makedirs(args.evaluation_output_dir)

    main(args)