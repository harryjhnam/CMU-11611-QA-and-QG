from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

ext_QA_model = "deepset/roberta-base-squad2"
gen_QA_model = "MaRiOrOsSi/t5-base-finetuned-question-answering"
boolean_classfier = "PrimeQA/tydiqa-boolean-question-classifier"
sim_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

AutoTokenizer.from_pretrained(ext_QA_model)
AutoModelForQuestionAnswering.from_pretrained(ext_QA_model)
AutoTokenizer.from_pretrained(gen_QA_model)
AutoModelWithLMHead.from_pretrained(gen_QA_model)
AutoTokenizer.from_pretrained(boolean_classfier)
AutoModelForSequenceClassification.from_pretrained(boolean_classfier)
