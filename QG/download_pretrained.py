#!/usr/bin/env python3

# downloading wordnet
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# downloaing pretrained models for pipeline 1
from keybert import KeyBERT
from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration
keyword_extraction_model = KeyBERT()
question_generation_tokenizer = PreTrainedTokenizerFast.from_pretrained('Sehong/t5-large-QuestionGeneration')
question_generation_model = T5ForConditionalGeneration.from_pretrained('Sehong/t5-large-QuestionGeneration')

# downloading pretrained models for pipeline 3
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
sum_tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
sum_model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')

# downloading pretrained models for filtering
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
model_id = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_id)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)