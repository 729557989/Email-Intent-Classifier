"""
    Description: configuration file regarding parameters and hyperparameters
    Author: Jimmy L.
    Date: Spring 2022
"""
import torch

# data file paths
data_path = "emails.csv"

# input column location
input_loc = "input"

# label column location
label_loc = "output"

# unwanted pandas columns
columns_to_drop =  ['date', 'from', 'to', 'cc', 'subject']

# paths for label tokenizer's dictionary
gp_dict_path = "models/model_general_para/label_dict/label_dict.pkl" # label tokenizer's dictionay file for model general para

# input tokenizer paths
gp_input_tknz_path = "models/model_general_para/input_tknz" # input tokenizer for model general para

# model checkpoint paths
general_para_BERT_path =  "models/model_general_para/bert/checkpoint.pth"

# training stat plot paths
gp_acc_plot_path = "models/model_general_para/training_plots/acc.png"
gp_loss_plot_path = "models/model_general_para/training_plots/loss.png"

# hyperparameters for the A.I. model
max_len = 200
vocab_size = 30522
batch_size = 6
torch.backends.cuda.matmul.allow_tf32 = True
split_size = 0.3 # 0.2 means 80% train data and 20% test data
rand_seed = 45
learning_rate = 2e-5 #2e-7

# number of labels
general_num_labels = 7 # should be 7, len(general_label_tokenizer)
specific_num_labels = 23 # should be 23, len(specific_label_tokenizer)

# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# default vocab.txt file for hugging face tokenizer
default_vocab = "models/default_vocab.txt"

# augmented data folder path
augmented_path = "Data_Augmentation/augmented"