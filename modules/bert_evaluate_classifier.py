import torch
import numpy as np
import time
import torch.nn as nn
from tqdm.notebook import tqdm
from modules.utils import set_seed, format_time, save_checkpoint


def evaluate_classifier(model, dataloader, device):
         
        model.eval()
        set_seed()
               
        with torch.no_grad():
            test_acc = 0
            test_loss=0
            predictions_list=[]
            num_samples=0
            prediction_probs_list=[]
            
            num_correct_preds=0
            labels_list=[]
            for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                # Load and feed data to model
                input_ids = batch[0].to(device)
                attention_masks = batch[1].to(device)
                labels = batch[2].to(device)
                
                outputs = model(input_ids, attention_masks)
                logits= outputs[0]
                
                predictions= torch.argmax(logits, dim=1)
                
                num_correct_preds+= torch.sum(predictions==labels.data)
                num_samples+=predictions.shape[0]

                predictions_list.append(predictions)
                prediction_probs_list.append(logits)
                labels_list.append(labels)
                torch.cuda.empty_cache()
                
        test_accuracy=num_correct_preds/num_samples
        performance={}
        performance['test_accuracy']=test_accuracy
        
        print(
            f' test accuracy: {test_accuracy:.6f}')
        
        return performance, torch.cat(predictions_list), torch.cat(labels_list)