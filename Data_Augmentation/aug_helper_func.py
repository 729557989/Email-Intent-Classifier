import os
import config
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import config
from textblob import TextBlob



# Functions for Data Augmentation and SAVING
def save_aug(target_save_labels, label_dfs, args, aug_model, grammar_model, n=8):
    """
    Purpose: Apply data augmentation and save all the augmeted csv into a dedicated folder

    Params:  1. target_save_labels (list):
                - A list of classes/labels to specify which type of data to augment

             2. label_dfs (dictionay):
                - A dictionary structured -> label : pandas dataframe

             3. args (TTSettings):
                - Parameters for the 5. grammar_model

             4. aug_model (ContextualWordEmbsAug):
                - The data augmentation model that augments the text

             5. grammar_model (HappyTextToText):
                - The grammar correction model that fixes the text's grammar

             6. n (integer):
                - How many instances to augment for every single instance of data

    Returns: Nothing
    """
    for label in target_save_labels:
        label_dfs[label] = label_dfs[label].drop_duplicates(keep="first")
        # max_instance = max_instance = int((df.shape[0] * aug_size)/len(label_tknz))
        # n = max_instance // label_dfs[label].shape[0]
        folder_path = "augmented"
        saving_path = (folder_path + "/" + label + ".csv")

        temp_df = pd.DataFrame(columns=['augmented_text', 'label'])
        for i in range(label_dfs[label].shape[0]):
            torch.cuda.empty_cache()
            print(f"Processing the {i}th text, total: {label_dfs[label].shape[0]}")
            text = label_dfs[label].iloc[i][config.input_loc]
            label = label_dfs[label].iloc[i][config.label_loc]

            augmented_text = aug_model.augment(text,n=n)
            for t in augmented_text:
                t = TextBlob(t)
                result = grammar_model.generate_text("grammar: " + str(t.correct()), args=args)
                temp_df = temp_df.append({
                    'augmented_text': result.text,
                    'label': label
                },ignore_index=True)
        temp_df.to_csv(saving_path)
        print(f"Successfully Saved Augmented DataFrame to {saving_path}")

# Data Augmentation functions for training
def load_dfs_from_folder(folder_path, in_path, out_path):
    """
    Purpose: Load all the augmented csv files from the input folder,
             and load them to a dictionary structured -> label : Pandas Dataframe

    Params:  1. folder_path (string):
                - Path of the folder storing the augmented csv files

             2. in_path (string):
                - The input column name of the augmented csv files

             3. out_path (string):
                - The label column name of the augmented csv files

    Returns: A dictionary structured -> label : Pandas Dataframe
    """
    dfs = {}
    for csv in os.listdir(folder_path):
        csv_path = folder_path + "/" + csv
        if csv_path != folder_path + "/note.md":
            temp_df = pd.read_csv(csv_path)
            temp_df = temp_df[[in_path, out_path]]
            temp_df = temp_df.rename(columns={
                in_path:  'input',
                out_path: 'output'
            })
            dfs[temp_df.iloc[0]['output']] = temp_df
    return dfs

def get_specific_label_dfs(raw_df, label_loc):
    """
    Purpose: Split the instances of data in raw_df based on specific labels/classes
             and load them to a dictionary structured -> label : Pandas Dataframe

    Params:  1. raw_df (Pandas Dataframe):
                - The df containing data

             2. label_loc (String):
                - The location where the output labels are stored in 1. raw_df

    Returns: A dictionary structured -> label : Pandas Dataframe
    """
    labels = list(raw_df[label_loc].unique())
    # a list of dataframes storing only instances of data belonging to one specific class/label
    label_dataframes = {}
    for label in labels:
        label_dataframes[label] = raw_df.loc[raw_df[label_loc] == label]
    return label_dataframes

def add_primitive_data(prim_dfs, aug_dfs, order_dict):
    """
    Purpose: Group the primitve data instances with their augmented ones

    Params:  1. prim_dfs (dictionary):
                - dictionary containing the augmented data structured -> label : Pandas Dataframe

             2. aug_dfs (dictionary):
                - dictionary containing the primitive data structured -> label : Pandas Dataframe

             3. order_dict (dictionary):
                - dictionary to decided on where to insert the primitive data every nth row,
                  structured -> label : integers

    Returns: A dictionary structured -> label : Pandas Dataframe
    """
    label_dataframes = {}
    for i, (label, instance) in enumerate(order_dict.items()):
        aug = list(aug_dfs[label].input)
        add = list(prim_dfs[label].input)
        res = []
        for i in range(len(aug)):
            res.append(aug[i])
            if i%instance  == 0:
                res.append(add[i//instance])
        
        dict = {
            "input": res,
            "output": [label for i in range(len(res))]
        }
        temp_df = pd.DataFrame(data=dict)
        label_dataframes[label] = temp_df
    return label_dataframes

def augmentation_train_test_split(dfs, order_dict, test_size, rand_seed):
    """
    Purpose: unique train test split for the group of augmented and primitive data

    Params:  1. dfs (dictionary):
                - A dictionary of Pandas DataFrame structure -> label : Pandas DataFrame

             2. order_dict (dictionary):
                - dictionary to decided on how to group every Pandas Dataframe to list
                  structured -> label : list

             3. test_size (integer):
                - The split size

             4. rand_seed (integer):
                - The seed for random shuffling

    Returns: list of training and validation data
    """
    batches = []
    for i, (label, df) in enumerate(dfs.items()):
        df_batch = np.array_split(df, df.shape[0]//order_dict[label])
        batches = batches + df_batch
    train, test = train_test_split(batches, test_size=test_size, random_state=rand_seed)

    train_instances = 0
    for list in train:
        train_instances+=len(list)
    test_instances = 0
    for list in test:
        test_instances+=len(list)
    test_percentage = float(test_instances)/float(train_instances)

    print("Note that the split size is lossy...\nCurrent distribution: Train -> {}%, Test -> {}%".format(
        int((1.0 - test_percentage)*100),
        int(test_percentage*100)
    ))
    return train, test