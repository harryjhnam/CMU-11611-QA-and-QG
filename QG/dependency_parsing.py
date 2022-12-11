#!/usr/bin/env python3
import re
import spacy
from spacy.tokens import Token
import nltk
from nltk.corpus import wordnet as wn

nlp = spacy.load("en_core_web_sm")
lemm = lambda token: token.lemma_
Token.set_extension('lemm', getter=lemm, force=True)

def sentence2question(txt):
    doc = nlp(txt)
    POS_elements = [t.pos_ for t in doc]
    qa_pair_lst = []

    # Only consider a sentence with no pronouns in it.
    if 'PRON' in POS_elements:
        return None

    if 'AUX' in POS_elements:
        # Init a question
        for token in doc:
            if token.pos_ == 'AUX':
                aux_token = token 
                question = aux_token.text.capitalize()
                break
        # Generate a question
        for token in doc:
            # Omit AUX
            if token == aux_token:
                continue

            question += ' '
            # If the first token is not a proper noun, lower the first letter before the inversion.
            if token.i == 0 and token.pos_ != 'PROPN':
                question += token.text.lower()
            else:    
                question += token.text
        
        question = question[:-2]
        question += '?'
        qa_pair_lst.append((question, 'yes'))

    elif 'VERB' in POS_elements:
        '''
        "VBD": verb, past
        "VBP": verb, non-3rd person singular present ;eg. I like soccer.
        "VBZ": verb, 3rd person singular present ;eg. Donggeun likes soccer.
        '''
        verb_to_aux = {'VBD':'Did', 'VBP':'Do', 'VBZ':'Does'}
        # Init a question

        is_verb = False
        for token in doc:  
            if token.tag_ in verb_to_aux.keys():
                verb_token = token
                question = verb_to_aux[token.tag_]
                is_verb = True
                break
        if not is_verb:
            return None

        # Check antonyms for the verb
        antonyms = []
        for syn in wn.synsets(verb_token.text):
            for i in syn.lemmas():
                if i.antonyms():
                    antonyms.append(i.antonyms()[0].name())
        if antonyms:
            antonym = antonyms[0]
            for t in nlp(antonym):
                antonym = t._.lemm
            question2 = question+''
        

        # Generate 'Yes' question
        for token in doc:
            question += ' '
            # If the first token is not a proper noun, lower the first letter before the inversion.
            if token.i == 0 and token.pos_ != 'PROPN':
                question += token.text.lower()
            # Lemmatize the verb.
            elif token == verb_token or token.pos_ == 'VERB':
                question += token._.lemm
            else:
                question += token.text    
        question = question[:-2]
        question += '?'
        qa_pair_lst.append((question, 'yes'))

        # Generate 'No' question
        if antonyms:
            for token in doc:
                question2 += ' '
                # If the first token is not a proper noun, lower the first letter before the inversion.
                if token.i == 0 and token.pos_ != 'PROPN':
                    question2 += token.text.lower()
                # Lemmatize the verb.
                elif token != verb_token and token.pos_ == 'VERB':
                    question2 += token._.lemm
                # Add the antonym
                elif token == verb_token:
                    question2 += antonym
                else:
                    question2 += token.text
            question2 = question2[:-2]
            question2 += '?'
            qa_pair_lst.append((question2, 'no'))

    else:
        return None

    # If the Yes question has a number in it, add one to make it false.
    if any(char.isdigit() for char in question):
        m = re.search(r"\d", question)
        idx = m.start()
        val = int(question[idx])
        new_val = val + 1
        question = question.replace(str(val), str(new_val))
        answer = 'no'
        qa_pair_lst.append((question, answer))

    return qa_pair_lst