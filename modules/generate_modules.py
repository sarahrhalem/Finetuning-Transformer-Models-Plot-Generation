import torch
import random
from tqdm.notebook import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from modules.utils import set_seed


def generate_text(model, tokenizer, device, num_samples, input_genres=None):
    # Generate plot samples based on Top-k and Top-p filtering and sampling methods available with HuggingFace model.generate method

    # set_seed() # Note: Seed is set in test method, enabling seed here impacts random choice of input genre
    generated_plots = []
    model.eval()
    with torch.no_grad():

        bos_tkn = tokenizer.bos_token
        sep_tkn = tokenizer.sep_token
        eos_tkn = tokenizer.eos_token

        # Input prompt for text generation is a randomly selected genre from our genre list.
        genre_list = ["romance", "drama", "comedy", "documentary", "action", "international",
                      "children", "crime", "horror", "anime", "other"]
        # Select random genre or user specified genre
        for i in range(num_samples):
            if input_genres is None:
                genre = random.choice(genre_list)
            else:
                genre = input_genres[i]

            prompt = bos_tkn + genre + ": " + sep_tkn
            prompts = (torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)).to(device)

            #
            samples = model.generate(prompts,  # input genre
                                     do_sample=True,
                                     top_k=50,
                                     min_length=15,
                                     max_length=50,
                                     top_p=0.95,
                                     num_return_sequences=1,
                                     repitition_penalty=1.1,
                                     no_repeat_ngram_size=2,
                                     temperature=1.1
                                     )

            generated_plot = "{}".format(tokenizer.decode(samples[0], skip_special_tokens=True, ))
            # print(len(samples))
            # print(generated_plot)
            generated_plots.append(generated_plot)

    return generated_plots


# Evaluates the test dataset on the fine-tuned model. Calculates the BLEU score based on generated samples from the model vs input
# plot samples.

def test_generate(model, tokenizer, dataloaders, device):
    model.eval()
    set_seed()

    with torch.no_grad():
        test_loss = 0
        plots = []
        references = []

        for step, batch in tqdm(enumerate(dataloaders['test_dataloader']), total=len(dataloaders['test_dataloader'])):

            input_ids = batch.to(device)
            outputs = model(input_ids, labels=input_ids)
            loss, logits = outputs[:2]
            batch_loss = loss.item()
            test_loss += batch_loss

            # List of genre decoded from each input_id
            genre_list = []
            # Pass input_id to references for BLEU score comparison with generated samples
            for input_id in input_ids:  #
                reference = tokenizer.decode(input_id, skip_special_tokens=True)
                x = reference.split()[0]
                references.append([reference])
                genre_list.append(x)

            # Generate sample using the same input genre as input_id
            generate = generate_text(model, tokenizer, device, num_samples=len(input_ids), input_genres=genre_list)
            plots += generate

            torch.cuda.empty_cache()

        # Smoothing function for BLEU score
        cc = SmoothingFunction()

        # bleu_score1= corpus_bleu(references, plots, weights=(1,0,0,0),smoothing_function=cc.method1)
        # BLEU score with default settings
        bleu_score_default = corpus_bleu(references, plots, smoothing_function=cc.method1)

        # BLEU score with modified weights to penalize higher n-gram precision
        bleu_score_modified1 = corpus_bleu(references, plots, weights=(0.5, 0.25, 0.25, 0),
                                           smoothing_function=cc.method1)
        bleu_score_modified2 = corpus_bleu(references, plots, weights=(0.5, 0.5), smoothing_function=cc.method1)

        # Loss and perplexity results
        mean_test_loss = test_loss / len(dataloaders['test_dataloader'])
        mean_test_perplexity = torch.exp(torch.tensor(mean_test_loss))

        print(
            f'test loss: {mean_test_loss:.6f}, test ppl: {mean_test_perplexity:.6f}, bleu score default:{bleu_score_default}, bleu score modified 1: {bleu_score_modified1}, bleu score modified 2: {bleu_score_modified2}')

        # Save test_performance
        test_performance = {'mean_test_loss': mean_test_loss, 'mean_test_perplexity': mean_test_perplexity,
                            'bleu_score_default': bleu_score_default, 'bleu_score_modified1': bleu_score_modified1,
                            'bleu_score_modified2': bleu_score_modified2}
    return test_performance, references, plots
