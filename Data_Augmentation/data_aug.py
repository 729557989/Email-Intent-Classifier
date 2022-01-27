"""
    Description: functions for text preprocessing
    Author: Jimmy L.
    Date: Spring 2022
"""
import sys
sys.path.append(r"C:\Users\45323\Desktop\新python文件夹\AI_Camp\AICS_Bert")

from aug_helper_func import get_specific_label_dfs, save_aug
import pandas as pd
from text_preprocessing import clean_df, map_labels
import config
import nlpaug.augmenter.word as naw
from textblob import TextBlob
from Label2Id import Label2Id
from happytransformer import HappyTextToText, TTSettings



if __name__ == '__main__':
    # NOTE: set up pretrained models
    happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    args = TTSettings(num_beams=5, min_length=1)
    try:
        aug = naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased',
            action="substitute",
            device="cuda"
        )
    except:
         aug = naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased',
            action="substitute",
            device="cpu"
        )

    # NOTE: load label tokenizer
    label_tknz = Label2Id()
    label_tknz.load_dict(config.gp_dict_path)
    
    # NOTE: Load and preprocess the primitive data to Pandas Dataframe
    df = pd.read_csv(config.data_path)
    df = clean_df(df, config.input_loc, config.columns_to_drop)
    df[config.label_loc] = df[config.label_loc].apply(map_labels)

    # NOTE: Get the dictionary structured -> label : Pandas Dataframe
    label_dfs = get_specific_label_dfs(df, config.label_loc)

    # NOTE: set up how many instances of data to augment for each class/label
    order_dict = {
        "monetary_issues" : 12,
        "program_info" : 4,
        "unactionable" : 3,
        "registration" : 8,
        "program_logistics" : 11,
        "scholarship" : 9,
        "others" : 8
    }

    # NOTE: augment and save data
    for idx, (label, aug_instance) in enumerate(order_dict.items()):
        print("Augmenting Data: {}".format(label))
        save_aug([label], label_dfs, args, aug, happy_tt, n=aug_instance)