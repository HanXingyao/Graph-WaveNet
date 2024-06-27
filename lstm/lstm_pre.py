import torch
import torch.nn as nn
from lstm1 import LSTMModel, TimeSeriesDataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def process_input_data(enhanced_data):
    enhanced_data = torch.tensor(enhanced_data, dtype=torch.float).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        outputs = model(enhanced_data)
        predicted = (outputs > 0.5).float()
    return predicted.cpu().numpy()

if __name__ == '__main__':
    pth_file = 'lstm_model.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size=1, hidden_size=64).to(device)
    model.load_state_dict(torch.load('lstm/2_lstm_model.pth'))
    model.eval()
    
    input_data0 = [47.83957760346365, 49.619772676153644, 51.25443050835863, 96.69933845966848,
                    99.89819460476217, 102.78303982685082, 105.29271728443604, 107.37235413851698,
                    108.97534892855055, 110.06510306233824, 154.6078763992814, 156.56742200425185]
    #[0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
    input_data1 = [51.25443051,52.70786889,53.94717943,54.94346222,55.67294461, 56.11792363,
                    56.26748004, 56.11792363,55.67294461,54.94346222,97.938649,98.65888406
                    ]
    # [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
    input_data2 = [
        98.65888406,141.9301186,100.8944774,103.5125222,105.7376963,107.5219105,
        108.8257925,109.620124,109.8869244,109.620124,108.8257925,107.5219105
        ]
    # [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 1.]
    for i in range(3):
        input_data = eval(f'input_data{i}')
        output_data = process_input_data(input_data)
        print("Processed output data:", output_data)


    data_path = 'lstm/lstm_train_data.csv'

    correct_predictions = 0
    total_predictions = 0
    for batch in DataLoader(TimeSeriesDataset(data_path), batch_size=1, shuffle=False):
        enhanced_data = batch['enhanced_data']
        original_data = batch['original_data']
        outputs = model(enhanced_data)
        predicted = (outputs > 0.5).float()

        predicted_ones = predicted == 1
        total_predictions += predicted_ones.sum().item()

        correct_predictions += (predicted_ones & (original_data == 1)).sum().item()
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
    else:
        accuracy = 0 
    print(f"precision: {accuracy:.4f}")
        # print("Predicted:", predicted.cpu().numpy())
        # print("Original:", original_data.cpu().numpy())
