"""
    Description: config file for hyperparameters
    Author: Jimmy L. @ AI - Camp
    Date: Spring 2022
"""
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import tqdm
# disable HuggingFace Warnings
import logging
logging.disable(logging.WARNING)
logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore')



def flat_accuracy(preds, labels):
    """
    Purpose: get the flat accuracy given predictions and ground truth labels

    Params:  1. preds (numpy array):
                - a 2d list of model prediction, shape: [batch_size, num_labels]

             2. labels (numpy array):
                - a 1d list of ground truth labels, shape [batch_size]

    Returns: an accuracy score (numpy.float64) with the given preds and labels
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



def get_classification_report(model, device, data_loader, label_tknz, get_acc=False):
    """
    Purpose: get an sklearn classification report as an evaluation metric for your model
             or, evaluate how well your model predicts with all of its multi-classes
    Params:  1. model (BertForSequenceClassification):
                - The AI model

             2. device (torch.device):
                - The device you would like to use, usually 'cpu' or 'cuda'

             3. data_loader (torch.utils.data.dataloader.DataLoader):
                - Input 'train_datalodaer' to get classification report for training set
                - Input 'valid_datalodaer' to get get classification report for validation set

             4. label_tknz (Label2ID):
                - Label tokenizer that convert string labels into integers

             5. get_acc (boolean):
                - If False, return sklearn's classification report
                - If True, return a list containing accuracy for every class/label that the model can predict

    Returns: If get_acc is True: Accuracy metric for every class (label) the your model can predict
             If get_acc is False: return an sklearn classification report
    """
    # set model to evaluation mode
    model.eval()

    # set model to utilize device
    model.to(device)
    predictions = []
    ground_truths = []

    for idx, batch in tqdm.tqdm(enumerate(data_loader), ascii=True):
        # cast input_seqs, attention_maks, labels in batch to device
        input_seqs = batch[0].to(device)
        attention_masks = batch[1].to(device)
        labels = batch[2].to(device)

        # Detach any tensors with gradients that are currently attached to the computational graph.
        with torch.no_grad():
            
            # Forward propagation/make predictions
            result = model(input_seqs, 
                           token_type_ids=None, 
                           attention_mask=attention_masks,
                           labels=labels,
                           return_dict=True)

            # get predictions and append ground_truth labels
            logits = result.logits
            y_pred = [np.argmax(pred.cpu().numpy()) for pred in logits]
            y_true = list(labels.cpu().numpy())
            predictions = predictions + y_pred
            ground_truths = ground_truths + y_true

    if get_acc == True:
        # Get and return accuracy score for every class (label) the your model can predict
        report = []
        labels = list(label_tknz.dict.keys())
        for label in labels:
            target = label_tknz.encoder(label)
            total_targets = float(ground_truths.count(target))
            num_correct = 0.0
            for i in range(len(ground_truths)):
                if ground_truths[i] == target:
                    if predictions[i] == target:
                        num_correct += 1.0
            try:
                acc = num_correct/total_targets
            except:
                acc = 0
            report.append('Accuracy for class "{}": {}'.format(label, str(acc*100)+'%'))
        return report
    else:
        # return an sklearn classification report
        return classification_report(ground_truths,
                                     predictions,
                                     target_names=list(label_tknz.dict.keys()) if label_tknz != None else None)



def plot_training_stats(stats, metric='loss', path=None):
    """
    Purpose: plot stats such as loss and accuracy

    Params:  1. stats (list):
                - List of dictionaries containing training stats

             2. metric (string):
                - The Y label for the plot, 'loss' or 'acc' (accuracy)

             3. path (string):
                - The path to save the plots

    Returns: displays a matplotlib.pyplot plot
    """
    # load stats into Pandas Dataframe
    pd.set_option('precision', 2)
    df = pd.DataFrame(data=stats)
    df = df.set_index('epoch')
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')
    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)
    # Plot the learning curve.

    # Determine the Y-labels
    if metric == 'loss':
        plt.plot(df['Training Loss'], 'b-o', label="Training")
        plt.plot(df['Valid. Loss'], 'g-o', label="Validation")
    else:
        plt.plot(df['Training Accur.'], 'b-o', label="Training")
        plt.plot(df['Valid. Accur.'], 'g-o', label="Validation")
    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss") if metric=='loss' else plt.ylabel("Accuracy")
    plt.legend()
    plt.xticks([i+1 for i in range(df.shape[0])])

    # save the plot png if path exists
    if path != None:
        plt.savefig(path)
    return plt.show()



def eval(model, device, data_loader, batch_size):
    """
    Purpose: Evaluate the A.I. model by returning an accuracy score and loss

    Params:  1. model (BertForSequenceClassification):
                - The AI model

             2. device (torch.device):
                - The device the A.I. model utilizes, usually 'cpu' or 'cuda'

             3. data_loader (torch.utils.data.dataloader.DataLoader):
                - The data_loader containing the data for the A.I. model to run predictions

             4. batch_size (integer):
                - Batch_size, or number of training examples utilized in one iteration

    Returns: The average eval accuracy (integer) and the average eval loss (integer)
    """
    total_loss = 0
    total_accuracy = 0
    # set model to evaluation mode
    model.eval()

    # set model to utilize device
    model.to(device if device != None else 'cpu')

    for idx, batch in tqdm.tqdm(enumerate(data_loader),
                                total=len(data_loader.dataset)/batch_size,
                                ascii=True,
                                desc="Eval"):
        
        # cast input_seqs, attention_maks, labels in batch to device
        input_seqs = batch[0].to(device if device != None else 'cpu')
        attention_masks = batch[1].to(device if device != None else 'cpu')
        labels = batch[2].to(device if device != None else 'cpu')

        # Detach any tensors with gradients that are currently attached to the computational graph.
        with torch.no_grad():
            
            # Forward propagation/make predictions
            output = model(input_seqs, 
                           token_type_ids=None, 
                           attention_mask=attention_masks,
                           labels=labels,
                           return_dict=True)

        # get and accumulate loss
        loss = output.loss
        total_loss += loss.item()

        logits = output.logits
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_accuracy += flat_accuracy(logits, label_ids)

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_accuracy / len(data_loader)
    avg_train_loss = total_loss / len(data_loader)
    return avg_val_accuracy, avg_train_loss
    


def plot_confusion_matrix(model, device, data_loader, get_acc=False):
    """
    Purpose: Plot an sklearn confusion matrix

    Params:  1. model (BertForSequenceClassification):
                - The AI model

             2. device (torch.device):
                - The device the A.I. model utilizes, usually 'cpu' or 'cuda'

             3. data_loader (torch.utils.data.dataloader.DataLoader):
                - The data_loader containing the data for the A.I. model to run predictions

             4. get_acc (boolean):
                - Whether to show percentage distributions for each grid in the confusion matrix

    Return:  Plot an confusion matrix
    """
    # set to model to evualuation mode
    model.eval()

    # set the model to utilize device
    model.to(device)
    predictions = []
    ground_truths = []
    for idx, batch in tqdm.tqdm(enumerate(data_loader),
                                ascii=True):

        # cast input_seqs, attention_maks, labels in batch to device       
        input_seqs = batch[0].to(device)
        attention_masks = batch[1].to(device)
        labels = batch[2].to(device)

        # Detach any tensors with gradients that are currently attached to the computational graph.
        with torch.no_grad():

            # Forward propagation/make predictions
            result = model(input_seqs, 
                            token_type_ids=None, 
                            attention_mask=attention_masks,
                            labels=labels,
                            return_dict=True)
            
            # Compute accuracy scores
            logits = result.logits
            y_pred = [np.argmax(pred.cpu().numpy()) for pred in logits]
            y_true = list(labels.cpu().numpy())
            predictions = predictions + y_pred
            ground_truths = ground_truths + y_true
    
    # get confusion matrix (numpy 2d array)
    cf_matrix = confusion_matrix(ground_truths, predictions)

    # plot the confusion matrix with seaborn
    if get_acc == True:
        return sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
    return sns.heatmap(cf_matrix, annot=True)