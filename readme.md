1. Folder -> "conda_config" contains:
    - bunch of .yml files I used, you may load them to use as conda environments
      NOTE: you could also pip install via. requirements.txt, but I preferr the .yml method of loading dependencies

2. Folder -> "Data_Augmentation" contains:
    - A folder containing csvs of augmented data
    - Has python scripts of how to augment + save data, and training with these data
    - Has an note.md that should tell you about how I augmented data

3. Folder -> "deployment" contains:
    - Has python scripts necessary for chrome plugin or making predictions on hosted web
    - Has a readme.md inside it for instructions

4. Folder -> "models" contains:
    - a folder structured:
        - bert (folder):
            .pth file containing a saved model with optimizer, LrScheduler and etc...

        - input_tknz (folder):
            files stored as the input tokenizer's weights,
            check out dataloader.py for more info about the input tokenizer

        - label_dict (folder):
            .pkl file containing a dictionary for the label tokenizer,
            checkout Label2ID.py for more info about the label tokenizer

        - training_plots (folder):
            stores files of training stats, such as loss and accuracy history plots if saved

5. File -> "config.py" contains:
    - bunch of configuration variables used throughout the folders for organization

6. File -> "dataloader.py" contains:
    - bunch of code & functions for prepping datasets before AI model training

7. File -> "eval.py" contains:
    - bunch of code & functions for evaluation methods

8. File -> "Label2ID.py" contains:
    - bunch of code & a class for creating a label tokenizer that serves similarily as Keras Tokenizer

9. File -> "run_train.py" contains:
    - a sample run of how to save label, input tokenizer, train and save an AI model, and other function sample runs.
      (But make sure you uncomment and read the COMMENTS)

10. File -> "text_preprocessing.py" contains:
    - a python script containing bunch of functions for cleaning the dataset

11. File -> "train.py" contains:
    - a python script containing functions for AI model training/fitting and saving

12. File -> "transformer.py" contains:
    - a python script containing a class for organization the AI model,
      and other fucntions to use such as making predictions, init optimizer, init LrScheduler and etc...