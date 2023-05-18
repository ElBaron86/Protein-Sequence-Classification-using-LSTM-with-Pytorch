"""
Functions to train models or use them to make predictions
"""

## Importations
import torch
import torchmetrics
import numpy as np
import time
from tqdm.auto import tqdm
from typing import Tuple, Dict, List



### Training function

def train(model : torch.nn.Module,
          loss_fn : torch.nn.Module,
          device : torch.device,
          train_loader : torch.utils.data.DataLoader,
          valid_loader : torch.utils.data.DataLoader,
          n_epochs : int,
          patience : int,
          save_dir : str = None,
          save_name : str = None,
          n_classes : int = None,
          optimizer : torch.optim.Optimizer = None) -> Dict[str, List]:
    
    """Train a model for a binary or a multiclass classification task, you have to specify
    'binary' as classification task argument if your data contains only 2 classes 

    Args:
        model : model instancied based from models.py file
        classification_task : 'multiclass' or 'binary', default 'multiclass' 
        optimizer : default Adam optimizer
        loss_fn : 
        device : 'cuda' or 'cpu', default = 'cpu'
        train_loader : DataLoader who loads train dataset with a specific collate funcion
        valid_loader : DataLoader who loads validation dataset with a specific collate funcion
        n_classes : number of classes on your data
        n_epochs : number of epochs to train
        patience : number of epochs to wait if valid loss doesn't improve
        save_dir : directory to save model state_dict
        save_name : name of model.state_dict to save in save_path    
    """
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())
        
    if n_classes > 2:
        acc_fn = torchmetrics.Accuracy(task = 'multiclass', num_classes = n_classes)
    else :
        acc_fn = torchmetrics.Accuracy(task = 'binary')
        
        
        
    results = { "train_loss": [],
                "train_acc": [],
                "valid_loss": [],
                "valid_acc": []
                }
        
    #loss_fn = loss_fn.to(device)
    acc_fn = acc_fn.to(device)
    
    start_time = time.time()
    
    # params for early stopping
    
    best_loss = np.inf
    best_epoch = 0

    # train with early stopping method, a good way to train a model without saving an outfitted state_dict !!!
    
    for epoch in range(n_epochs):
            
        # train metrics
        l, a = 0, 0
        
        model.train()
        
        for sequences, labels, lengths in tqdm(train_loader):
            
            # inputs to device
            labels = labels.to(device).long()    # labels must be in torch.Long type
            sequences = sequences.to(device)
            
            # zero_grad()
            optimizer.zero_grad()
            
            # Forward pass
            preds = model(sequences, lengths).squeeze()
            # get the predicted label
            pred_label = preds.argmax(dim=1)
            
            # calculate the loss
            loss = loss_fn(preds, labels)
            
            # calculate accuracy
            acc = acc_fn(pred_label, labels)
            
            # backpropagation
            loss.backward()
            
            # optimisation step
            optimizer.step()
            
            # store all epoch losses & acc
            l+=loss.item()
            a+=acc.item()
            
        # store average epoch loss & acc
        results["train_loss"].append(l/len(train_loader))
        results["train_acc"].append(a/len(train_loader))
                
                
        # Evaluate model on validation set
        model.eval()
        with torch.no_grad():
            ll, aa = 0, 0
            for sequences, labels, lengths in tqdm(valid_loader):
                labels = labels.to(device).long()               # labels must be in torch.Long type
                sequences = sequences.to(device)
                
                # Forward pass
                preds = model(sequences, lengths).squeeze()
                
                # get the predicted label
                pred_label = preds.argmax(dim=1)
                
                ll += loss_fn(preds, labels).item()
                aa += acc_fn(pred_label, labels).item() 

        results["valid_loss"].append(ll/len(valid_loader))
        results["valid_acc"].append(aa/len(valid_loader))
        
        ################### check if metrics are better : Early stopping ########################
        
        if results['valid_loss'][-1] < best_loss: # if eval loss is better, save the state dict else whait for max patience epochs
            best_loss = results['valid_loss'][-1]
            best_epoch = epoch
            if save_dir is not None:
                torch.save(model.state_dict(), save_dir+'/'+save_name+'.pt')
            
        else:   # break the train to prevent overfit
            if epoch - best_epoch >= patience:
                print("!!!")
                print(f"Early stopping at epoch : {epoch+1}")
                break
        
        print("*"*50)
        print(f"Epoch : {epoch+1}")
        print(f"Train --> loss : {results['train_loss'][-1]} | acc {results['train_acc'][-1]}")
        print(f"Evaluation --> loss {results['valid_loss'][-1]} | acc {results['valid_acc'][-1]}")
        print("*"*50)
        
    # running time
    end_time = time.time()
    total_time = end_time - start_time
    
    print('~~'*30)
    print(f"Training takes {total_time} seconds")
    print('~~'*30)
    
    if save_dir is not None:
        
        # save metrics to visualize training curves
        np.save(save_dir+"/"+save_name+"_train_losse", np.array(results["train_loss"]))
        np.save(save_dir+"/"+save_name+"_train_accuracy", np.array(results["train_acc"]))
        np.save(save_dir+"/"+save_name+"_valid_losse", np.array(results["valid_loss"]))
        np.save(save_dir+"/"+save_name+"_valid_accuracy", np.array(results["valid_acc"]))

    return results
    
## Test function to predict a class

def test(model : torch.nn.Module,
         loader : torch.utils.data.DataLoader,
         device : torch.device) -> Dict[np.array, np.array]:
    
    """function to make prediction(s) with the trained model

    Returns:
        dict with true_labels and predicted_labels as np.array
    """
    results = {"y_true" : [],
               "y_pred" : []}
    
    model.eval()

    with torch.no_grad():

        for sequences, labels, lengths in tqdm(loader):
            labels = labels.to(device)
            sequences = sequences.to(device)
            
            # Forward pass
            preds = model(sequences, lengths).squeeze()
            
            # get predicted
            pred_label = preds.argmax(dim=1)
            
            results['y_true'].append(labels.cpu().numpy())    
            results['y_pred'].append(pred_label.cpu().numpy())
        
        results["y_true"] = np.array(results["y_true"])
        results["y_pred"] = np.array(results["y_pred"])
        
        ### concatenate all
        results["y_true"] = np.concatenate(results["y_true"], axis=0)
        results["y_pred"] = np.concatenate(results["y_pred"], axis=0)
        
                
    return results
