Hi There! :)



1. I will update pretrained model weights in the folder: 'model'
2. Configurations stored at file: 'config.py', becareful with pathings at this file
3. You can check the descriptions @ file: transformer.py's predict function for how the model predicts,
   But... I should also have descripted this @ file make_pred.py
4. Check out requirements.txt for required dependencies (including version)
5. There's no need to edit anything @ files: Label2ID.py, text_preprocessing.py, and transformer.py

6. (1) go to NLP_MODEL/main.py, uncomment the commented code to train a model
   (2) make sure it's saved to "models\model_general_para\bert"
   (3) Copy that "checkpoin.pth" file to: "deployment\model\weights"
   (4) Congratulations! You may Now go to "deployment\make_pred.py" and run a prediction!