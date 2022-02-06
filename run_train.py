"""
    Description: A sample script that uses every functionality of the files in this folder
        1. Using Label2ID and Saving it's fitted dictionary
        2. Creating a new BertTokenizer, Saving it, and Loading it
        3. Load label_tknz + input_tknz, get processed dataloaders
        4. Get label distributions for processed dataloaders
        5. Initialize the model, optimizer, and scheduler for learning rate
        6. Model Training
        7. Get the sklearn classification report
        8. Plot the traing stats (loss, acc)
        9. Make predictions with the saved model
"""
from eval import plot_training_stats
import config
from transformer import BertUncased, predict
from transformers import BertTokenizer
import pandas as pd
from text_preprocessing import map_labels
from Label2Id import Label2Id
from dataloader import EmailsDataset, dataloader, get_label_distribution
from train import training
from eval import get_classification_report


if __name__ == '__main__':
    # # NOTE: Using Label2ID and Saving it's fitted dictionary
    # label_tknz = Label2Id()
    # df = pd.read_csv(config.data_path)
    # df = df.drop(config.columns_to_drop, axis=1)
    # df[config.label_loc] = df[config.label_loc].apply(map_labels)
    # labels_list = list(df[config.label_loc])
    # label_tknz.fit(labels_list)
    # label_tknz.build_vocab(labels_list)
    # label_tknz.save_dict(config.gp_dict_path)
    # print("Dictionary Saved")



    # # NOTE: Creating a new BertTokenizer, Saving it, and Loading it
    # input_tknz = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True,max_len=config.max_len)
    # input_tknz.save_pretrained(config.gp_input_tknz_path)
    # print("Input_tknz Save")
    # input_tknz = BertTokenizer.from_pretrained(config.gp_input_tknz_path)



    # # NOTE: Load label_tknz + input_tknz, get processed dataloaders
    # epochs = 10
    # label_tknz = Label2Id()
    # label_tknz.load_dict(config.gp_dict_path)
    # input_tknz = BertTokenizer.from_pretrained(config.gp_input_tknz_path)
    # emails_data = EmailsDataset(config.input_loc, config.label_loc, config.max_len,
    #                             input_tknz, label_tknz,
    #                             from_df = False, df = None, path = config.data_path,
    #                             preprocess=True, drop_redundant=True, rand_seed=config.rand_seed,
    #                             do_specific_labels=False)
    # train_dataloader, valid_dataloader = dataloader(emails_data, config.batch_size, config.split_size)
    # print("train_dataloader and valid_dataloader is ready!")


    # # NOTE: Get label distributions for processed dataloaders
    # print(get_label_distribution(train_dataloader, label_tknz))
    # print(get_label_distribution(valid_dataloader, label_tknz))



    # # NOTE initialize the model, optimizer, and scheduler for learning rate
    # bert = BertUncased()
    # bert.build_model(len(label_tknz))
    # bert.build_optimizer(config.learning_rate)
    # bert.build_LRscheduler(train_dataloader, epochs)


    # # NOTE: Model Training
    # training_stats = training(bert.model, epochs, config.batch_size,
    #                           config.device, bert.optimizer, bert.lrScheduler,
    #                           train_dataloader, valid_dataloader,
    #                           saving_path=config.general_para_BERT_path,
    #                           save_best_weights=True,
    #                           model_class=BertUncased(),
    #                           comparison_metric='acc', data_compared='valid',
    #                           transfer_learning=False,
    #                           get_training_stats=True,
    #                           new_learining_rate=True,
    #                           patience=5)



    # # NOTE: after training, you can get the sklearn classification report
    # print(get_classification_report(bert.model, config.device, valid_dataloader, label_tknz, get_acc=False))



    # # NOTE: after training, you can plot the traing stats (loss, acc), could also save the png
    # plot_training_stats(training_stats, path=config.gp_loss_plot_path)
    # plot_training_stats(training_stats, metric='acc', path=config.gp_acc_plot_path)


    # NOTE: how to make predictions with your pretrained model
    label_tknz = Label2Id()
    label_tknz.load_dict(config.gp_dict_path)
    input_tknz = BertTokenizer.from_pretrained(config.gp_input_tknz_path)
    bert = BertUncased()
    bert.build_model(len(label_tknz))
    bert.load_weights(config.general_para_BERT_path, config.device)
    sent = ["I would like to speak to ai camp regarding my child's learning status",
            "When can we expect a refund?",
            "Is the course still available on December 2021?"]
    pred = predict(sent, bert.model, input_tknz, label_tknz, device='cpu')
    print(pred)
