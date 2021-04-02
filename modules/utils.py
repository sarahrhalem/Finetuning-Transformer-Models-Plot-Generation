import torch
import numpy as np
import csv
import random
import pickle
import matplotlib.pyplot as plt


def save_checkpoint(state, checkpoint_path):
    print("Saving checkpoint ... ")
    torch.save(state, checkpoint_path)
    print("Checkpoint:", checkpoint_path, "saved.")


def load_checkpoint(model, optimizer, scheduler, load_checkpoint_path):
    print("Loading checkpoint ... ")
    checkpoint = torch.load(load_checkpoint_path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, scheduler, start_epoch


# Time and seed util methods
def format_time(start_time, end_time):
    hours, remainder = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds))


def set_seed():
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    torch.cuda.manual_seed_all(10)


# function to map listings to genres by dictionary key
def map_function(dictionary):
    def my_map(x):
        res = ""
        for key in dictionary.keys():
            if x in dictionary[key]:
                res = key
                break
        return res

    return my_map


# Plotting methods for training/validation

# Combined plots train/val

def plot_train_val_loss(history, title):
    fig = plt.figure()
    epoch_count = range(1, len(history['train_loss']) + 1)
    plt.plot(epoch_count, (history['train_loss']), label='Training Loss')
    plt.plot(epoch_count, (history['val_loss']), label='Validation Loss')
    plt.ylim(0, 2.5)
    plt.xticks(epoch_count)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.show()
    return fig


def plot_train_val_perplexity(history, title):
    fig = plt.figure()
    epoch_count = range(1, len(history['train_perplexity']) + 1)
    plt.plot(epoch_count, (history['train_perplexity']), label='Train perplexity')
    plt.plot(epoch_count, (history['val_perplexity']), label='Validation perplexity')
    plt.xticks(epoch_count)
    plt.ylim(0, 12)
    plt.xlabel("epochs")
    plt.ylabel("perplexity")
    plt.title(title)
    plt.legend()
    plt.show()
    return fig


# Individual plots for train and val to visualise to scale

def plot_train_loss(history, title):
    fig = plt.figure()
    epoch_count = range(1, len(history['train_loss']) + 1)
    plt.plot(epoch_count, (history['train_loss']), label='Training Loss')
    plt.ylim(0, 1)
    plt.xticks(epoch_count)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.show()
    return fig


def plot_val_loss(history, title):
    fig = plt.figure()
    epoch_count = range(1, len(history['val_loss']) + 1)
    plt.plot(epoch_count, (history['val_loss']), label='Validation Loss')
    plt.xticks(epoch_count)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.show()
    return fig


def plot_train_perplexity(history, title):
    fig = plt.figure()
    epoch_count = range(1, len(history['train_perplexity']) + 1)
    plt.plot(epoch_count, (history['train_perplexity']), label='Train perplexity')
    plt.xticks(epoch_count)
    plt.xlabel("epochs")
    plt.ylabel("perplexity")
    plt.title(title)
    plt.legend()
    plt.show()
    return fig


def plot_val_perplexity(history, title):
    fig = plt.figure()
    epoch_count = range(1, len(history['val_perplexity']) + 1)
    plt.plot(epoch_count, (history['val_perplexity']), label='Validation perplexity')
    plt.xticks(epoch_count)
    plt.xlabel("epochs")
    plt.ylabel("perplexity")
    plt.title(title)
    plt.legend()
    plt.show()
    return fig


def plot_train_accuracy(history, title):
    fig = plt.figure()
    epoch_count = range(1, len(history['train_accuracy']) + 1)
    plt.plot(epoch_count, (history['train_accuracy']), label='Train accuracy')
    plt.xticks(epoch_count)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title(title)
    plt.legend()
    plt.show()
    return fig


def plot_val_accuracy(history, title):
    fig = plt.figure()
    epoch_count = range(1, len(history['val_accuracy']) + 1)
    plt.plot(epoch_count, (history['val_accuracy']), label='Validation accuracy')
    plt.xticks(epoch_count)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title(title)
    plt.legend()
    plt.show()
    return fig


def plot_train_val_accuracy(history, title):
    fig = plt.figure()
    epoch_count = range(1, len(history['train_accuracy']) + 1)
    plt.plot(epoch_count, (history['train_accuracy']), label='Train accuracy')
    plt.plot(epoch_count, (history['val_accuracy']), label='Validation accuracy')
    plt.xticks(epoch_count)
    plt.ylim(0, 12)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title(title)
    plt.legend()
    plt.show()
    return fig


# Methods to save results and save plot samples to csv

def save_test_results(results, filename):
    test_performance = results[0]
    with open(filename, 'wb') as handle:
        pickle.dump(test_performance, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


def load_test_results(filename):
    with open(filename, 'rb') as handle:
        test_performance = pickle.load(handle)
    return test_performance


def plot_samples_csv(generated_plots, plot_samples_path):
    input_list = zip(generated_plots)
    with open(plot_samples_path, "w", newline="") as myfile:
        write = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for row in input_list:
            write.writerow(row)
    return
