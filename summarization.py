from datasets import load_dataset
from transformers import BartForConditionalGeneration, AutoTokenizer
import csv
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("a1noack/bart-large-gigaword")
model = BartForConditionalGeneration.from_pretrained("a1noack/bart-large-gigaword")

dataset = load_dataset("gigaword", split="test")
documents = dataset['document'][:4]
summaries = dataset['summary'][:4]

with open('hw5_output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Sentence', 'Summary', 'Model Output', 'Confidence', 'Input Length'])

# function from Debarati Das
def calculate_perplexity_and_likelihood(scores): 
    likelihood = 0 
    perplexity = 0 
    probabilities = [] # Logit is normalized already 
    for score in scores[0]: 
        logit = score.numpy() 
        # Based on the document, since logit is normalized prob is simply np.exp(logit) 
        prob = np.exp(logit) 
        probabilities.append(prob) 
    likelihood = np.sum(np.log(probabilities)) 
    perplexity = np.exp(-likelihood / len(probabilities)) 
    return perplexity, likelihood

def generate_sentence(description, input_ids, do_sample=False, num_beams=1, top_k=50, top_p=1):
    output = model.generate(input_ids, do_sample=do_sample,
                                      num_beams=num_beams, top_k=top_k,
                                      top_p=top_p, min_length=0, max_new_tokens=60,
                                      pad_token_id=tokenizer.eos_token_id,
                                      output_scores=True,
                                      return_dict_in_generate=True)
    input_length = input_ids.shape[1]
    tokens = output.sequences[0]
    # logits = output.scores

    scores = model.compute_transition_scores(
        output.sequences, output.scores, normalize_logits=True
    )
    (perp, likelihood) = calculate_perplexity_and_likelihood(scores)

    sentence = tokenizer.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    s = "".join(sentence)
    print(f"{description}: {s}")
    return s, likelihood

for (document, summary) in zip(documents, summaries):
    print(f"\n----------------\n{document}")
    print(f"Summary: {summary}\n")

    input_ids = tokenizer(document, return_tensors="pt", truncation=True, max_length=128,padding=True)['input_ids']
    summary_tokens = tokenizer(summary, return_tensors="pt").input_ids

    # beam search (num_beams > 1, do_sample = False)
    # https://huggingface.co/blog/how-to-generate suggests 5 beams
    sentence, likelihood = generate_sentence("beam search 5 beams", input_ids, do_sample=False, num_beams=5)

    with open('output.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([document, summary, sentence, likelihood, input_ids.shape[1]])
