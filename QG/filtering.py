from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

def filtering(questions_list, top_n, threshold):
    """
    Compute the perplexity score for the questions
    :param questions_list: list of questions
    :return: list of perplexity scores
    """
    max_length = model.config.n_positions
    stride = 1
    result = []
    for question in questions_list:
        encodings = tokenizer([question], return_tensors="pt")
        seq_len = encodings.input_ids.size(1)
        prev_end_loc = 0
        nlls = []
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        result.append(ppl.item())
    perplexity_scores = [(x, y) for x, y in zip(questions_list, result) if y < threshold]
    perplexity_scores.sort(key=lambda x: x[1])
    perplexity_sorted_questions = [x[0] for x in perplexity_scores][:top_n]
    return perplexity_sorted_questions
