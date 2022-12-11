#!/usr/bin/env python3

import torch

from keybert import KeyBERT
from transformers import PreTrainedTokenizerFast
from transformers import T5ForConditionalGeneration

import string

keyword_extraction_model = KeyBERT()
question_generation_tokenizer = PreTrainedTokenizerFast.from_pretrained('Sehong/t5-large-QuestionGeneration')
question_generation_model = T5ForConditionalGeneration.from_pretrained('Sehong/t5-large-QuestionGeneration')


def pipeline1(paragraphs, keyword_confidence_threshold=0.7):
    
    qa_pairs = []

    for paragraph in paragraphs:

        # extract keywords
        keywords = keyword_extraction_model.extract_keywords(paragraph, keyphrase_ngram_range=(1, 2))
        unique_keywords = set([k for k, s in keywords])

        for keyword, score in keywords:
            if score < keyword_confidence_threshold:
                continue

            if keyword not in unique_keywords:
                continue
            unique_keywords.remove(keyword)

            # generate question
            qg_input_ids = question_generation_tokenizer.encode(f'answer:{keyword} content:{paragraph}')
            qg_input_ids = [question_generation_tokenizer.bos_token_id] + qg_input_ids + [question_generation_tokenizer.eos_token_id]
            question_ids = question_generation_model.generate(torch.tensor([qg_input_ids]))

            decode = question_generation_tokenizer.decode(question_ids.squeeze().tolist(), skip_special_tokens=True)
            decode = decode.replace(' # # ', '').replace('  ', ' ').replace(' ##', '')

            # exclue "question: " or "? " prefix
            if decode[:10] == "question: ":
                question = decode[10:]
            elif decode[:2] == "? ":
                question = decode[2:]
            else:
                question = decode

            if len(question) == 0:
                continue
            
            if question[-1] != '?':
                if question[-1] in string.punctuation:
                    question = question[:-1]
                question += '?'

            qa_pairs.append((question, keyword))

    return qa_pairs