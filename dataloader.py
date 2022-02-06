"""
    Description: create dataloader for our data
"""
from torch.utils.data import Dataset, DataLoader, random_split
from text_preprocessing import clean_df, map_labels
import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split

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
   - input_tknz = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True,max_len=max_len)

2. You can SAVE your input_tknz:
   - input_tknz.save_pretrained(path)

3. You can LOAD your saved input_tknz:
   - input_tknz = BertTokenizer.from_pretrained(path)

NOTE: special methods with input_tknz
convert the sequence back to word sentence by passing in a 1d list of sequence:
    input_tknz.convert_ids_to_tokens(sequence)
"""

class EmailsDataset(Dataset):
    """
    A Dataset class for cleaning, organizing, and storing data
    """
    def __init__(self, input_loc, label_loc, max_len,
                 input_tknz, label_tknz,
                 from_df = False, df = None, path = None,
                 preprocess=True, drop_redundant=True, rand_seed=None,
                 do_specific_labels=False):
        # check config.py for configured parameters
        if from_df == True:
            assert df.equals(None) == False
        if path == None:
            assert from_df == True
        """
        Params:  1. input_loc (string):
                    - The column location of input data in the Pandas DataFrame from: pandas.read_csv(8. path)

                 2. label_loc (string):
                    - The column location of labels in the Pandas DataFrame from: pandas.read_csv(8. path)

                 3. max_len (integer):
                    - The MAX sequence length for each input text,
                      or how many Dimensions you want to represent each text

                 4. input_tknz (transformers.models.bert.tokenization_bert.BertTokenizer):
                    - An input tokenizer thats CONVERTS the string text a list of pytorch float

                 5. label_tknz (Label2Id):
                    - An label/output tokenizer that CONVERTS the string labels to integer

                 6. from_df (boolean):
                    - Whether to load data from Pandas Dataframe,
                      an alternative from providing a path to read_csv such as (8. path)
                    - NOTE: 6. from_df works inconjunction with 8. df, because df can't equal None if this is True

                 7. df (Pandas Dataframe):
                    - The Pandas DataFrame as input if you would like to load data through Pandas Dataframes,
                      an alternative from providing a path to read_csv such as (8. path)
                    - NOTE: 7. from_df works inconjunction with 8. df, only if 7. from_df is True you provide a df
                
                 8. path (string):
                    - The path of the CSV so pandas.to_csv will read a comma-separated values (csv) file into DataFrame
                    - NOTE: This is ONLY used when 6. from_df is equal to None

                 9. preprocess (boolean):
                    - Whether or not to apply the clean_df function from text_preprocessing.py to the self.raw_df(pandas DataFrame)
                
                10. drop_redundant (boolean):
                    - If True, drop redundant rows in self.df

                11. rand_seed (integer):
                    - Do random shuffling with a random seed to create reproducible shuffling

                12. do_specific_labels (boolean):
                    - If False:  map the labels in self.df(pandas DataFrame) into general labels

        Returns: An class object with 3 values to unpack when using the Python enumerate function:
                 tokenized inputs (2 dimensional), shape:[length of dataset, max_len]
                 attention masks  (2 dimensional), shape:[length of dataset, max_len]
                 labels           (1 dimensional), shape:[length of dataset]
        """
        # initialize neecessary self variables
        self.input_loc = input_loc
        self.label_loc = label_loc
        self.max_len = max_len
        self.input_tknz = input_tknz
        self.label_tknz = label_tknz

        # get pandas Dataframe by reading the csv file from path
        self.raw_df = pd.read_csv(path) if from_df == False else df

        # clean the Pandas Dataframe self.raw_df with the clean_df function from file text_preprocessing.py
        self.df = clean_df(self.raw_df, input_loc) if preprocess == True else self.raw_df
        self.df = self.df[[self.input_loc, self.label_loc]]
        self.df = self.df.drop_duplicates(keep="first") if drop_redundant == True else self.df
        
        # Map labels at self.df's labels location to general labels if do_specific_labels was set to False
        if do_specific_labels == False:
            self.df[label_loc] = self.df[label_loc].apply(map_labels)

        # Do random shuffling if rand_seed doesn't equal None
        if rand_seed != None:
            # Do random shuffling if rand_seed doesn't equal None
            self.df = self.df.sample(n=len(self.df),random_state=rand_seed)

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


def dataloader(dataset, batch_size, split_size, valid_dataset=None):
    """
    Purpose: Construct a dataloader for training used in PyTorch
    
    Params:  1. dataset (EmailsDataset):
                - The dataset storing input and output data

             2. batch_size (integer):
                - Batch_size, or number of training examples utilized in one iteration

             3. split_size (integer):
                - The split size for the data, say for 0.25, 75% will be training data and 25% validation data
            
             4. valid_dataset (EmailsDataset):
                - The validation dataset to provide if you decide to skip train test split,
                  then the function only takes in dataset(train) and valid_dataset(valid) and spit out the PyTorch DataLoader for them

    Returns: train_dataloader (torch.utils.data.dataloader.DataLoader) containing training data
             valid_dataloader (torch.utils.data.dataloader.DataLoader) containing validation data
    """
    if valid_dataset == None:
        # do train test split with SkLearn
        train_dataset, valid_dataset = train_test_split(dataset, test_size=split_size, shuffle=False)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    else:
        train_dataloader = DataLoader(dataset, batch_size=batch_size)
        
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    # return train and valid DataLoader 
    return train_dataloader, valid_dataloader



def get_label_distribution(dataloader, label_tknz):
    """
    Purpose: Count instance of data for every label in the dataloader

    Params:  1. dataloader (torch.utils.data.dataloader.DataLoader):
                - The dataloader containing the data

             2. label_tknz (Label2Id object):
                - Label tokenizer that converts string labels into integers.

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