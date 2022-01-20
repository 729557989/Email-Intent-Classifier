import config
from transformer import predict

"""
Prediction test run
NOTE: It took my laptop (cuda 11.5, rtx3060 @60W) 8.3 second to make one prediction
"""


if __name__ == '__main__':
    # NOTE: single prediction
    sent = ["what are qualifications"]
    # # NOTE: multi predictions
    # sent = [
    #     "Please give me my *Y*&#@*$&( refund",
    #     "OK, thank you",
    #     "Hey, what's the summer program about?"
    # ]
    pred = predict(sent, config.bert.model, config.input_tknz, config.label_tknz, device=config.device)
    print(pred)