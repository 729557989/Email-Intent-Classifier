"""
    Description: functions for text preprocessing
    Author: Jimmy L. @ AI - Camp
    Date: Spring 2022
"""
import sys
root = r"C:\Users\45323\Desktop\新python文件夹\AI_Camp\AICS_Bert"
sys.path.append(root)

import config
import pandas as pd
from aug_helper_func import load_dfs_from_folder, get_specific_label_dfs, add_primitive_data, augmentation_train_test_split
from text_preprocessing import clean_df, map_labels
from Label2Id import Label2Id
from transformers import BertTokenizer
from dataloader import EmailsDataset, get_label_distribution, dataloader
from transformer import BertUncased
import config
from train import training

if __name__ == '__main__':
    # NOTE: Load Data
    aug_dfs = load_dfs_from_folder(config.augmented_path, 'augmented_text', 'label')
    df = pd.read_csv(config.data_path)
    df = clean_df(df, config.input_loc, config.columns_to_drop)
    df[config.label_loc] = df[config.label_loc].apply(map_labels)
    prim_dfs = get_specific_label_dfs(df, config.label_loc)

    order_dict = {
        "monetary_issues" : 12,
        "program_info" : 4,
        "unactionable" : 3,
        "registration" : 8,
        "program_logistics" : 11,
        "scholarship" : 9,
        "others" : 8
    }

    concat_dfs = add_primitive_data(prim_dfs, aug_dfs, order_dict)

    order_dict = {
        "monetary_issues" : 13,
        "program_info" : 5,
        "unactionable" : 4,
        "registration" : 9,
        "program_logistics" : 12,
        "scholarship" : 10,
        "others" : 9
    }

    train, test = augmentation_train_test_split(concat_dfs, order_dict, 0.22, config.rand_seed)
    train_df = pd.concat(train, axis=0)
    valid_df = pd.concat(test, axis=0)
    
    label_tknz = Label2Id()
    label_tknz.load_dict(config.gp_dict_path)
    input_tknz = BertTokenizer.from_pretrained(config.gp_input_tknz_path)

    train_data = EmailsDataset(config.input_loc, config.label_loc, config.max_len,
                           input_tknz, label_tknz,
                           from_df = True, df = train_df, path = None,
                           preprocess=True, drop_redundant=True, rand_seed=config.rand_seed,
                           do_specific_labels=False)
    valid_data = EmailsDataset(config.input_loc, config.label_loc, config.max_len,
                            input_tknz, label_tknz,
                            from_df = True, df = valid_df, path = None,
                            preprocess=True, drop_redundant=True, rand_seed=config.rand_seed,
                            do_specific_labels=False)
    
    train_dataloader, valid_dataloader = dataloader(
        train_data,
        config.batch_size,
        config.split_size,
        valid_dataset=valid_data
    )

    # NOTE: Get class/label distributions in the dataset
    print(get_label_distribution(train_dataloader, label_tknz))
    print(get_label_distribution(valid_dataloader, label_tknz))

    # NOTE: Set up the AI model and hyperparameters
    epochs = 15
    bert = BertUncased()
    bert.build_model(len(label_tknz))
    bert.build_optimizer(config.learning_rate)
    bert.build_LRscheduler(train_dataloader, epochs)

    # NOTE: Training and Saving!
    training_stats = training(bert.model, epochs, config.batch_size,
                            config.device, bert.optimizer, bert.lrScheduler,
                            train_dataloader, valid_dataloader,
                            saving_path=config.general_para_BERT_path,
                            save_best_weights=True,
                            model_class=BertUncased(),
                            comparison_metric='loss', data_compared='valid',
                            transfer_learning=False,
                            get_training_stats=True,
                            new_learining_rate=False,
                            patience=4)