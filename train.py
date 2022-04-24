"""
    Description: Containing functions for model training.
    Author: Jimmy L. @ AI - Camp
    Date: Spring 2022
"""
from eval import flat_accuracy, eval
import torch
import tqdm
# Disable HuggingFace Warning
import logging
logging.disable(logging.WARNING)
logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore')


def training(model, epochs, batch_size, device, optimizer, scheduler,
             train_dataloader, valid_dataloader, saving_path=None, save_best_weights=False,
             model_class=None, comparison_metric='loss', data_compared='valid',
             transfer_learning=True, get_training_stats=False, 
             new_learining_rate=False, patience=3):
    """
    Purpose: Train the A.I. Model and save it

    Params:  1. model (BertForSequenceClassification):
                - The A.I. for training

             2. epochs (integer):
                - Number of training epochs

             3. batch_size (integer):
                - Batch_size, or number of training examples utilized in one iteration

             4. device (torch.device):
                - What device to use for A.I. training, generally 'cpu' or 'cuda'

             5. optimizer (transformers.optimization.AdamW):
                - The optimizer for the A.I. model to reduce loss

             6. scheduler (torch.optim.lr_scheduler.LambdaLR):
                - The scheduler that lowers learning rate throughout training

             7. train_dataloader (torch.utils.data.dataloader.DataLoader):
                - The dataloader containing the train data for training

             8. valid_dataloader (torch.utils.data.dataloader.DataLoader):
                - The dataloader containing the validation data for training

             9. saving_path (string):
                - The path where the model's weights as stored, so called a checkpoint

            10. save_best_weights (string):
                - Whether or not save only the model with the best weights through training
                - NOTE: this functionality should be used in conjucntion with:
                        12. comparison_metric, 13. data_compared

            11. model_class (BertUncased):
                - The class object for torch.save() to save the model's weights

            12. comparison_metric (string):
                - What metric to utilized to determine if a model has the best weights for 10. save_best_weights
                - This could be either 'loss' (loss), or 'acc' (accuracy)
                - NOTE: this functionality should be used in conjucntion with:
                        12. comparison_metric, 13. data_compared

            13. data_compared (string):
                - What data is utilized to determine if a model has the best weights for 10. save_best_weights
                - This could be either 'train' (training data), or 'valid' (validation data)
                - NOTE: this functionality should be used in conjucntion with:
                        12. comparison_metric
                        13. data_compared

            14. transfer_learning (boolean):
                - Whether or not to train the Saved model in the saving_path
                - NOTE: must provide the saving_path to use this functionality

            15. get_training_stats (boolean):
                - Whether to return a list of training stats, such as loss & accuracy for plotting
            
            16. new_learining_rate(boolean):
                - Whether to continue using the saved optimizer and LrScheduler settings from the checkpoint at 9. saving_path

            17. patience (integer):
                - The number of epochs to wait before early stop if the model is considered having no progress,
                  this determined by 12. comparison_metric & 13. data_compared.
                - NOTE: this functionality should be used in conjucntion with:
                        10. save_best_weights
                        12. comparison_metric
                        13. data_compared

    Returns: a list of training stats (refer to 15.) if get_training_stats is set to True, or else return nothing
    """
    assert comparison_metric == 'loss' or comparison_metric == 'acc'
    assert data_compared == 'valid' or comparison_metric == 'train'
    if transfer_learning == True:
        assert saving_path != None

    # Cast the A.I. model to utilize device
    model.to(device)
    training_stats = []
    patience_counter = 0

    # Epoch training loops
    for epoch in range(1, epochs+1):        
        if save_best_weights==True and saving_path != None:
            # Clear unused GPU memories
            torch.cuda.empty_cache()

            # Load the checkpoint weights from saving_path
            if epoch > 1 or transfer_learning==True:
                checkpoint = torch.load(saving_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                if new_learining_rate == False and epoch == 1:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    scheduler.load_state_dict(checkpoint["LrScheduler_state_dict"])
                prev_metric = checkpoint[data_compared+'_'+comparison_metric]

                # Cast the A.I. model to utilize device
                model.to(device)

                # Clear unused GPU memories
                torch.cuda.empty_cache()

        # Reset computed accuracy and loss
        total_train_acc = 0
        total_train_loss = 0

        # Cast the model to training mode
        model.train()
        print("<"+"-"*80+">")

        print("Current Lr: {}, Initial_Lr: {}".format(
            optimizer.state_dict()['param_groups'][-1]['lr'],
            optimizer.state_dict()['param_groups'][-1]['initial_lr']
        ))
        
        # Make forward and backward propagations
        for idx, batch in tqdm.tqdm(enumerate(train_dataloader),
                                    total=len(train_dataloader.dataset)/batch_size,
                                    ascii=True,
                                    desc="Model Training"):
            input_seqs = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)

            # Clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()
            
            # Perform a forward propagation
            output = model(input_seqs, 
                           token_type_ids=None, 
                           attention_mask=attention_masks, 
                           labels=labels,
                           return_dict=True)
            
            # Calculate the accuracy of our predictions vs labels
            logits = output.logits
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            total_train_acc += flat_accuracy(logits, label_ids)

            # Accumulate training loss
            loss = output.loss
            total_train_loss += loss.item()

            # Back propagate to get gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient
            optimizer.step()
            
            # Update learning rates
            scheduler.step()
        
        # Compute average loss and accuracies
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_acc = total_train_acc / len(train_dataloader)
        avg_valid_acc, avg_valid_loss = eval(model, device, valid_dataloader, batch_size)

        # Save best weights happen here
        if epoch > 1 or transfer_learning == True:
            if data_compared == 'train' and comparison_metric == 'loss':
                curr_metric = avg_train_loss
            elif data_compared == 'train' and comparison_metric == 'acc':
                curr_metric = avg_train_acc
            elif data_compared == 'valid' and comparison_metric == 'loss':
                curr_metric = avg_valid_loss
            elif data_compared == 'valid' and comparison_metric == 'acc':
                curr_metric = avg_valid_acc
        
        if saving_path != None:
            if save_best_weights==True:
                # Prerequisities: If an saving path exist, save_best_weights is set to True

                # If this isn't transfer learning: always save the model during first epoch
                if epoch == 1 and transfer_learning==False:
                    save_checkpoint(
                        model, model_class,
                        avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc,
                        optimizer, scheduler, saving_path
                    )
                    print("Model Saved For First Epoch!")

                # If transfer_learning is set to True or this isn't the first epoch,
                #   then save the weights if the current model is considered as 'better' than the previous one.
                elif (comparison_metric=='loss' and (curr_metric < prev_metric)) or (comparison_metric=='acc' and (curr_metric > prev_metric)):
                    save_checkpoint(
                        model, model_class,
                        avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc,
                        optimizer, scheduler, saving_path
                    )
                    print("Weights Updated and Saved! {} {} improved from {} to {}".format(
                        data_compared, comparison_metric, prev_metric, curr_metric
                    ))
                    patience_counter = 0
                
                # If not, don't save the weights and move on to the next epoch.
                else:
                    print("Weights weren't updated")
                    patience_counter += 1
            
            # Always save the model if saving_path is set to True but not save_best_weights
            else:
                save_checkpoint(
                    model, model_class,
                    avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc,
                    optimizer, scheduler, saving_path
                )
                print("Model Saved")

        # Report progress with accuracy, loss metrics
        print(f"Epoch: {epoch}\nTrain Eval: Avg.Acc: {avg_train_acc}, Avg.Loss: {avg_train_loss}")
        print(f"Valid Eval: Avg.Acc: {avg_valid_acc}, Avg.Loss: {avg_valid_loss}")

        # If get_training_stats is set to True, append the stats at the end of every epoch
        if get_training_stats == True:
            training_stats.append(
                {
                 'epoch':           epoch,
                 'Training Loss':   avg_train_loss,
                 'Training Accur.': avg_train_acc,
                 'Valid. Loss':     avg_valid_loss,
                 'Valid. Accur.':   avg_valid_acc
                }
            )
        
        # If the model haven't improve since an 'patience' amount of epochs, end training early
        if patience != None and patience_counter == patience:
            print(f"Model haven't improved for {patience_counter} epochs, Early Stopping Now")

             # Return the list of training stats if get_training_stats is set to True
            if get_training_stats == True:
                return training_stats
            else:
                return

    # Return the list of training stats if get_training_stats is set to True
    if get_training_stats == True:
        return training_stats



def save_checkpoint(model, model_class,
                    train_loss, valid_loss, train_acc, valid_acc,
                    optimizer, scheduler, saving_path):
    """
    Purpose: Save model weights

    Params:  1. model (BertForSequenceClassification):
                - The A.I. model

             2. model_class (BertUncased):
                - The class_object where the 1. model was stored at

             3. train_loss (float):
                - The A.I. model's training loss

             4. valid_loss (float):
                - The A.I. model's validation loss

             5. train_acc (float):
                - The A.I. model's training accuracy

             6. valid_acc (float):
                - The A.I. model's validation accuracy

             7. optimizer (transformers.optimization.AdamW):
                - The A.I. model's optimizer object

             8. scheduler (torch.optim.lr_scheduler.LambdaLR):
                - The A.I. model's scheduler object

             9. saving_path (string):
                - The path to save the weights, AKA the checkpoint

    Returns: Nothing
    """
    checkpoint = {'model_object': model_class,
                  'train_loss': train_loss,
                  'valid_loss': valid_loss,
                  'train_acc': train_acc,
                  'valid_acc': valid_acc,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict':optimizer.state_dict(),
                  'LrScheduler_state_dict':scheduler.state_dict()}
    torch.save(checkpoint, saving_path)