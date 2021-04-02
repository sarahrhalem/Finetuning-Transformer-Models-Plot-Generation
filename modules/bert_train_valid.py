import torch
import numpy as np
import time
import torch.nn as nn
from tqdm.notebook import tqdm
from modules.utils import set_seed, format_time


# Training/Validation method
def bert_train_val(model, dataloaders, starting_epoch, optimizer, scheduler, epochs, device):
    print("\n\n" + "-" * 15)
    print("| TRAINING... |")
    print("-" * 15)
    set_seed()
    start_training_time = time.time()

    # Define running history for train and val
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    # Training loop
    for epoch in range(starting_epoch, epochs):
        train_loss = 0
        train_acc = 0
        model.train()
        for step, batch in tqdm(enumerate(dataloaders['train_dataloader']), total=len(dataloaders['train_dataloader'])):
            # Load and feed data to model
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(input_ids, labels=labels, attention_mask=attention_masks)
            loss = outputs.loss
            logits = outputs.logits

            batch_loss = loss.item()
            train_loss += batch_loss
       
            logits = logits.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()

            predictions = np.argmax(logits, axis=1).flatten()
            # labels = labels.flatten()

            correct = 0
            for i in range(0, len(predictions)):
                if predictions[i] == labels[i]:
                    correct = correct + 1
            batch_accuracy = correct / len(labels)
            train_acc += batch_accuracy

            if step % 100 == 0:
                print("Epoch: ", epoch + 1, "/", epochs, "Batch: ", step + 1, "/", len(dataloaders['train_dataloader']),
                      "Loss: ", train_loss / (step + 1), "Accuracy: ", batch_accuracy)

            loss.backward()
            # Apply gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Optimzer/Learning rate schedular step
            optimizer.step()
            scheduler.step()

            torch.cuda.empty_cache()

        # Loss and accuracy results by epoch
        end_epoch_time = time.time()
        epoch_train_accuracy = train_acc / len(dataloaders['train_dataloader'])
        epoch_train_loss = train_loss / len(dataloaders['train_dataloader'])
        epoch_train_time = format_time(start_training_time, end_epoch_time)
        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_accuracy)

        print(
            f' epoch: {epoch + 1}, train loss: {epoch_train_loss:.6f}, train accuracy: {epoch_train_accuracy:.6f}, train time:{epoch_train_time}')

        # Switch to evaluation mode and run validation
        print("Validating...")

        start_val_time = time.time()
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(dataloaders['val_dataloader']), total=len(dataloaders['val_dataloader'])):
                # Load and feed data to model
                input_ids = batch[0].to(device)
                attention_masks = batch[1].to(device)
                labels = batch[2].to(device)

                model.zero_grad()

                outputs = model(input_ids, labels=labels, attention_mask=attention_masks)
                loss = outputs.loss
                logits = outputs.logits

                batch_loss = loss.item()
                val_loss += batch_loss

                logits = logits.detach().cpu().numpy()
                labels = labels.to('cpu').numpy()

                predictions = np.argmax(logits, axis=1).flatten()

                correct = 0
                for i in range(0, len(predictions)):
                    if predictions[i] == labels[i]:
                        correct = correct + 1

                batch_accuracy = correct / len(labels)
                val_acc += batch_accuracy

                torch.cuda.empty_cache()
                end_val_time = time.time()

        epoch_val_time = format_time(start_val_time, end_val_time)
        epoch_val_loss = val_loss / len(dataloaders['val_dataloader'])
        epoch_val_acc = val_acc / len(dataloaders['val_dataloader'])
        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_acc)

        print(
            f' epoch: {epoch + 1}, val loss: {epoch_val_loss:.6f}, val accuracy: {epoch_val_acc:.6f}, val_time: {epoch_val_time}')

        # Record results to dictionary to return
        performance_history = {'train_loss': train_loss_history, 'val_loss': val_loss_history,
                               'train_accuracy': train_acc_history, 'val_accuracy': val_acc_history}

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
