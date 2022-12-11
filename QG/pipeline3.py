#!/usr/bin/env python3

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

from dependency_parsing import sentence2question

device = "cuda" if torch.cuda.is_available() else "cpu"
sum_tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
sum_model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum').to(device)


def summarize(text):
    if sum_tokenizer.unk_token_id in sum_tokenizer(text)["input_ids"]:
        return None
    batch = sum_tokenizer([text], truncation=True, padding="longest", return_tensors="pt").to(device)
    translated = sum_model.generate(**batch)
    summary = sum_tokenizer.batch_decode(translated, skip_special_tokens=True)
    return summary


def pipeline3(paragraphs):

    qa_pairs = []

    for paragraph in paragraphs:
        if '=' in paragraph or len(paragraph) < 100:
            continue
        summarization = summarize(paragraph)
        if summarization is not None:
            summary = summarization[0]
            if len(summary) < 30 or ':' in summary:
                continue

            # perform the rule-based question generation
            results = sentence2question(summary)
            
            if results is None:
                continue

            for question, answer in results:
                qa_pairs.append((question, answer))

    return qa_pairs
