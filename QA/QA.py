#!/usr/bin/env python3

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline,AutoModelWithLMHead,AutoModelForSequenceClassification
import numpy as np
from sentence_transformers import SentenceTransformer, util
import argparse

import warnings
warnings.filterwarnings('ignore')

ext_QA_model = "deepset/roberta-base-squad2"
gen_QA_model = "MaRiOrOsSi/t5-base-finetuned-question-answering"
boolean_classfier = "PrimeQA/tydiqa-boolean-question-classifier"
sim_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def get_window_paragraph(text, question , window_size = 3):
    paragraphs = [para for para in text.split("\n") if para if para != "\t\t"]
    #label = "the Gurjara-Pratihara dynasty."
    embedding_1 = sim_model.encode(question, convert_to_tensor=True)
    max_sim = 0
    for paragraph in paragraphs:
        paragraph_sim_list = []
        for sentence in paragraph.split(". "):
            embedding_2 = sim_model.encode(sentence, convert_to_tensor=True)
            a = util.pytorch_cos_sim(embedding_1, embedding_2)
            paragraph_sim_list.append(a.item())
            # print(sentence, a.item())
        if max(paragraph_sim_list) > max_sim:
            max_sim = max(paragraph_sim_list)
            max_paragraph = paragraph
            max_sentence_idx = np.argmax(paragraph_sim_list)
    low_bound = 0 if max_sentence_idx - window_size < 0 else max_sentence_idx - window_size
    window_paragraph = '. '.join(max_paragraph.split(". ")[low_bound:max_sentence_idx + window_size])
    return window_paragraph

def boolean_classify(question):
    boolean_tokenizer = AutoTokenizer.from_pretrained(boolean_classfier)
    boolean_classify = AutoModelForSequenceClassification.from_pretrained(boolean_classfier)
    boolean_classifier = pipeline('text-classification', model=boolean_classify, tokenizer=boolean_tokenizer)
    result = boolean_classifier(question)
    # print(question , result)
    return result[0]['label']

def ext_QA(question, window_paragraph):
    ext_tokenizer = AutoTokenizer.from_pretrained(ext_QA_model)
    ext_model = AutoModelForQuestionAnswering.from_pretrained(ext_QA_model)
    ext_QA = pipeline('question-answering', model=ext_model, tokenizer=ext_tokenizer)
    QA_input = {
        'question': question,
        'context': window_paragraph
    }
    result = ext_QA(QA_input)
    return result['answer']

def gen_QA(question, window_paragraph):
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_QA_model)
    gen_model = AutoModelWithLMHead.from_pretrained(gen_QA_model)

    input = f"question: {question} context: {window_paragraph}"
    encoded_input = gen_tokenizer([input],
                                  return_tensors='pt',
                                  max_length=512,
                                  truncation=True)
    output = gen_model.generate(input_ids=encoded_input.input_ids,
                                     attention_mask=encoded_input.attention_mask)
    output = gen_tokenizer.decode(output[0], skip_special_tokens=True)
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=''' Question Answering & Generation Pipeline''')
    parser.add_argument('--article_path', type=str, default = "documents/indian_states/Gujarat.txt")
    parser.add_argument('--questions_path', type=str, default = "questions.txt")
    args = parser.parse_args()

    # Extractive QA Model
    with open(args.article_path, "r") as f:
        text = f.read()
    with open(args.questions_path, "r") as f:
        questions = f.readlines()

    for question in questions:
        window_paragraph = get_window_paragraph(text, question)
        
        if boolean_classify(question) == "LABEL_0":
            # Extractive Question Model
            print(ext_QA(question, window_paragraph))
        else:
            # Generative Question Model
            print(gen_QA(question, window_paragraph))



