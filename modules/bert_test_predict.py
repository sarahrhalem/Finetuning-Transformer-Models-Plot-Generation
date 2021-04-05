
import torch
import random
from tqdm.notebook import tqdm
from modules.utils import set_seed

def evaluate_classifier(model, dataloaders, device):
         
        model.eval()
        set_seed()
        
        
        with torch.no_grad():
            test_acc = 0
            test_loss=0
            predictions=[]
            prediction_probs=[]
            num_correct_preds=0
            true_labels=[]
            for step, batch in tqdm(enumerate(dataloaders['test_dataloader']), total=len(dataloaders['test_dataloader'])):
                # Load and feed data to model
                input_ids = batch[0].to(device)
                attention_masks = batch[1].to(device)
                labels = batch[2].to(device)
                
                outputs = model(input_ids, attention_mask=attention_masks)
                loss, logits = outputs[:2]
                batch_loss = loss.item()
                test_loss += batch_loss

                _, predictions= torch.argmax(outputs, dim=1).to(device)
                
                num_correct_preds+= torch.sum(predictions==labels.data)
                        
                batch_accuracy = correct / len(labels)
                test_acc += batch_accuracy

                predictions.extend(predictions)
                prediction_probs.extend(outputs)
                true_labels.extend(labels)
                torch.cuda.empty_cache()
                
          
        test_loss = test_loss / len(dataloaders['test_dataloader'])
        test_acc = num_correct_preds / len(dataloaders['val_dataloader']) 
        predictions= torch.stack(predictions).cpu()
        prediction_probs= torch.stack(prediction_probs).cpu
        true_labels=torch.stack(labels).cpu()
        performance={}
        performance['test_loss']=test_loss
        performance['test_accuracy']=test_acc
        
        return performance, predictions, prediction_probs, true_labels
               
