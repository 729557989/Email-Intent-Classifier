"""
    Description: create dataloader for our data
    Author: Jimmy L. @ AI - Camp
    Date: Spring 2022
"""
from torch.utils.data import Dataset, DataLoader, random_split
from text_preprocessing import clean_df, map_labels
import pandas as pd
import torch

# disable HuggingFace Warning
import logging
logging.disable(logging.WARNING)
logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore')


"""
NOTE: for the input tokenizer (input_tknz)
first do: from transformers import BertTokenizer
1. You can CREATE a new one:
      input_tknz = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True,max_len=max_len)

2. You can SAVE your input_tknz:
      input_tknz.save_pretrained(path)

3. You can LOAD your saved input_tknz:
      input_tknz = BertTokenizer.from_pretrained(path)

NOTE: special methods with input_tknz
convert the sequence back to word sentence by passing in a 1d list of sequence:
    input_tknz.convert_ids_to_tokens(sequence)
"""

class EmailsDataset(Dataset):
    """
    A Dataset class for cleaning, organizing, and storing data
    """
    def __init__(self, path, input_loc, label_loc, max_len,
                 input_tknz, label_tknz,
                 preprocess=True, columns_to_drop=None, do_specific_labels=False):
        # check config.py for configured parameters
        """
        Params:  1. path (string):
                    The path of the csv data

                 2. input_loc (string):
                    The column location of input data in the csv/pandas dataframe

                 3. label_loc (string):
                    The column location of labels in the csv/pandas dataframe

                 4. max_len (integer):
                    The max sequence length for each input

                 5. input_tknz (transformers.models.bert.tokenization_bert.BertTokenizer):
                    The input tokenizer that transform string inputs into a list of pytorch float

                 6. label_tknz (Label2Id):
                    The label tokenizer that transforms string labels into integers

                 7. preprocess (boolean):
                    Whether or not to apply the clean_df function from text_preprocessing.py to the self.raw_df(pandas DataFrame)

                10. columns_to_drop (list of string):
                    Drop unwanted columns in self.raw_df(pandas DataFrame)

                11. do_specific_labels (boolean):
                    If False:  map labels in self.df(pandas DataFrame) into general labels

        Returns: An class object with 3 values to unpack:
                 tokenized inputs (2 dimensional), shape:[length of dataset, max_len]
                 attention masks  (2 dimensional), shape:[length of dataset, max_len]
                 labels           (1 dimensional), shape:[length of dataset]
        """
        self.input_loc = input_loc
        self.label_loc = label_loc
        self.max_len = max_len

        # get pandas Dataframe by reading the csv file from path
        self.raw_df = pd.read_csv(path)

        # clean the Pandas Dataframe self.raw_df with the clean_df function from file text_preprocessing.py
        self.df = clean_df(self.raw_df,
                           input_loc,
                           drop_cols=columns_to_drop) if preprocess == True else self.raw_df
        self.input_tknz = input_tknz
        self.label_tknz = label_tknz
        
        # Map labels at self.df's labels location to general labels if do_specific_labels was set to False
        if do_specific_labels == False:
            self.df[label_loc] = self.df[label_loc].apply(map_labels)

    def __getitem__(self, index):
        """ CLASS METHOD
        complete every necessary preprocessing methods here:
        1. Tokenize the (string) inputs
        2. Get attention mask for the input
        3. Convert labels to ids (integer)
        """

        text = self.df.iloc[index][self.input_loc]

        """
        "encode_plus" will:
        (1) Tokenize the sentence.
        (2) Prepend the `[CLS]` token to the start.
        (3) Append the `[SEP]` token to the end.
        (4) Map tokens to their IDs.
        (5) Pad or truncate the sentence to `max_length`
        (6) Create attention masks for [PAD] tokens
        """
        encoded_dict = self.input_tknz.encode_plus(
                        text,                         # Sentence to encode.
                        add_special_tokens = True,    # Add '[CLS]' and '[SEP]'
                        max_length = self.max_len,    # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True, # Construct attn. masks.
                        return_tensors = 'pt',        # Return pytorch tensors.
                   )

        input_seq = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']

        # convert the string into integer
        label = self.label_tknz.encoder(self.df.iloc[index][self.label_loc])

        # return the input sequence, attention mask, and label
        return input_seq[0], attention_mask[0], label
        

    def __len__(self):
        """ 
        NOTE:CLASS METHOD
        Return the number of instances of data in the EmailsDataset class object
        """
        return self.df.shape[0]

def dataloader(dataset, batch_size, split_size=0.8, random_seed=None):
    """
    Purpose: Construct a dataloader for training used in PyTorch
    
    Params:  1. dataset (EmailsDataset):
                The dataset storing input and output data

             2. batch_size (integer):
                Batch_size, or number of training examples utilized in one iteration

             3. split_size (integer):
                The split size for the data, say for 0.8, 80% will be training data and 20% validation data

             4. random_seed (integer):
                The random seed for random shuffling that can help random shuffling reproducable

    Returns: train_dataloader (torch.utils.data.dataloader.DataLoader) containing training data
             valid_dataloader (torch.utils.data.dataloader.DataLoader) containing validation data
    """

    # Divide the dataset by randomly selecting samples.
    train_size = int(split_size * len(dataset))
    valid_size = len(dataset) - train_size

    # Apply random seed if exist
    if random_seed != None:
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size],
                                                    generator=torch.Generator().manual_seed(random_seed))
    else:
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    
    # get train and valid DataLoader 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    # return train and valid DataLoader 
    return train_dataloader, valid_dataloader



def get_label_distribution(dataloader, label_tknz):
    """
    Purpose: Count instance of data for every label in the dataloader

    Params:  1. dataloader (torch.utils.data.dataloader.DataLoader):
                The dataloader containing the data

             2. label_tknz (Label2Id object):
                Label tokenizer that converts string labels into integers.

    Returns: A list containing string sentences about number of instance for every label existing in the dataloader
    """
    instance_of_labels = []
    for idx, (input_seq, attention_mask, label) in enumerate(dataloader):
        instance_of_labels.append(label)
    labels = []
    for batch in instance_of_labels:
        for ground_truths in batch:
            labels.append(label_tknz.decoder(int(ground_truths)))
    label_distributions = []
    for label in list(label_tknz.dict.keys()):
        label_distributions.append('label: "{}" has "{}" instances'.format(label, labels.count(label)))
    return label_distributions