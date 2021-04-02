import torch
import time
import torch.nn as nn
from tqdm.notebook import tqdm
from modules.utils import set_seed, format_time


# Training and validation method to fine-tune the model on the Netflix plot dataset
def gpt2_train_val(model, dataloaders, tokenizer, starting_epoch, optimizer, scheduler, epochs, device):
    print("\n\n" + "-" * 15)
    print("| TRAINING... |")
    print("-" * 15)
    set_seed()
    start_training_time = time.time()

    # Define running history for train and val
    train_loss_history = []
    val_loss_history = []
    train_perplexity_history = []
    val_perplexity_history = []

    # Training loop
    for epoch in range(starting_epoch, epochs):
        train_loss = 0
        model.train()
        for step, batch in tqdm(enumerate(dataloaders['train_dataloader']), total=len(dataloaders['train_dataloader'])):
            # Load and feed data to model
            input_ids = batch.to(device)
            model.zero_grad()
            outputs = model(input_ids, labels=input_ids)

            loss = outputs[0]
            batch_loss = loss.item()
            train_loss += batch_loss

            if step % 200 == 199:
                print("Epoch:", epoch + 1, "/", epochs, "Batch:", step + 1, "/", len(dataloaders['train_dataloader']),
                      "Loss", train_loss / 200)
                train_loss = 0.0

            # Generates a model output including special tokens in order to visualise the training process and model learning
            model.eval()
            if step % 100 == 0 and step != 0:
                samples = model.generate(  # decoder_start_token_id=50258,
                    bos_token_id=50257,
                    do_sample=True,
                    top_k=50,
                    max_length=50,
                    min_length=15,
                    top_p=0.95,
                    num_return_sequences=1,
                    repition_penalty=1.1,
                    no_repeat_ngram_size=2,
                    temperature=1.1
                )

                for i, sample in enumerate(samples):
                    print("{}".format(tokenizer.decode(sample, skip_special_tokens=False)))

            # Return to train mode and back propagate loss
            model.train()
            loss.backward()
            # Apply gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Optimizer/Learning rate scheduler step
            optimizer.step()
            scheduler.step()

            torch.cuda.empty_cache()

        # Loss and perplexity results by epoch
        end_epoch_time = time.time()
        epoch_train_loss = train_loss / len(dataloaders['train_dataloader'])
        epoch_train_perplexity = torch.exp(torch.tensor(epoch_train_loss))
        epoch_train_time = format_time(start_training_time, end_epoch_time)
        train_loss_history.append(epoch_train_loss)
        train_perplexity_history.append(epoch_train_perplexity)

        print(
            f' epoch: {epoch + 1}, train loss: {epoch_train_loss:.6f}, train ppl: {epoch_train_perplexity:.6f}, train time:{epoch_train_time}')

        # Switch to evaluation mode and run validation
        print("Validating...")

        start_val_time = time.time()
        model.eval()
        val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(dataloaders['val_dataloader']), total=len(dataloaders['val_dataloader'])):
                input_ids = batch[0].to(device)
                outputs = model(input_ids, labels=input_ids)
                loss = outputs[0]
                # loss, logits= outputs[:2] # outputs has two elements loss and logits
                batch_loss = loss.item()
                val_loss += batch_loss

                torch.cuda.empty_cache()
                end_val_time = time.time()

        epoch_val_time = format_time(start_val_time, end_val_time)
        epoch_val_loss = val_loss / len(dataloaders['val_dataloader'])
        epoch_val_perplexity = torch.exp(torch.tensor(epoch_val_loss))
        val_loss_history.append(epoch_val_loss)
        val_perplexity_history.append(epoch_val_perplexity)
        # print("Validation time: ", epoch_val_time)

        print(
            f' epoch: {epoch + 1}, val loss: {epoch_val_loss:.6f}, val ppl: {epoch_val_perplexity:.6f}, val_time: {epoch_val_time}')

        # Record results to dictionary to return
        performance_history = {'train_loss': train_loss_history, 'val_loss': val_loss_history,
                               'train_perplexity': train_perplexity_history, 'val_perplexity': val_perplexity_history}

        # Save model checkpoint at end of train_val run, also saves performance history
        if epoch == epochs - 1:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'performance_history': performance_history,
                'epoch': epoch + 1,
            }
        # save_checkpoint(checkpoint, f"./checkpoint_{checkpoint['epoch']}.pth.tar")
        print("")
        print("Training Finished")

    return performance_history
