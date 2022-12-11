#!/usr/bin/env python3

# For pipeline 1
from pipeline1 import pipeline1

# For pipeline 2
from pipeline2 import pipeline2

# For pipeline 3
from pipeline3 import pipeline3

# Filtering
from filtering import filtering

import argparse

import warnings
warnings.filterwarnings('ignore')


def main(args):

    # Read document
    with open(args.article_path, 'r') as f:
        text = f.read()
        paragraphs = [p for p in text.split('\n') if p != '']

    paragraphs = paragraphs[:5]

    # Generate Questions
    generated_questions = []

    """
        === The first QG pipeline ===
    """
    qa_pairs1 = pipeline1(paragraphs)
    generated_questions.extend( [q for q, a in qa_pairs1] )

    """
        === The second QG pipeline ===
    """
    qa_pairs2 = pipeline2(paragraphs)
    generated_questions.extend( [q for q, a in qa_pairs2] )

    """
        === The third QG pipeline ===
    """
    qa_pairs3 = pipeline3(paragraphs)
    generated_questions.extend( [q for q, a in qa_pairs3] )

    """
        === Get top n questions ===
    """
    top_n_questions = filtering(generated_questions, top_n=args.n_questions, threshold=300)

    for i in range(args.n_questions):
        if i >= len(top_n_questions):
            print()
        else:
            print(top_n_questions[i])
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--article_path", type=str, default="./data/documents/constellations/scorpius_constellation.txt",
                            help="input document file")
    parser.add_argument("--n_questions", type=int, default=100,
                            help="number of questions to generate")
    args = parser.parse_args()

    main(args)
