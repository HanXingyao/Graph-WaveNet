import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class TimeSeriesDataset(Dataset):
    def __init__(self, data_file, window_size=12):
        self.data = pd.read_csv(data_file)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.window_size
        enhanced_data = self.data.iloc[start_idx:end_idx]['enhanced_data'].values.astype(float)
        original_data = self.data.iloc[start_idx:end_idx]['original_data'].values.astype(float)
        return {
            'enhanced_data': torch.tensor(enhanced_data, dtype=torch.float).unsqueeze(-1).to(device),
            'original_data': torch.tensor(original_data, dtype=torch.float).to(device)
        }

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_rate=0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 12)  # 输出12个时间步的预测
        self.drop_out = nn.Dropout(dropout_rate)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.drop_out(lstm_out[:, -1, :])  # 只使用最后一个时间步的隐藏状态
        out = self.fc(out)
        return torch.sigmoid(out)  # 使用sigmoid激活函数以确保输出在0和1之间

model = LSTMModel(input_size=1, hidden_size=64).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 训练模型
def train_model(epochs):
    model.train()
    dataloader = DataLoader(TimeSeriesDataset('./lstm/lstm_train_data.csv'), batch_size=1, shuffle=True)
    for epoch in range(epochs):
        print(f'Epoch [{epoch+1}/{epochs}]')
        running_loss = 0.0
        for batch in dataloader:
            enhanced_data = batch['enhanced_data']
            original_data = batch['original_data']
            
            optimizer.zero_grad()
            outputs = model(enhanced_data)
            loss = criterion(outputs, original_data)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(dataloader)}')
    torch.save(model.state_dict(), './lstm/2_lstm_model.pth')

def deploy_model():
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(TimeSeriesDataset('./lstm/lstm_train_data.csv'), batch_size=1, shuffle=False):
            enhanced_data = batch['enhanced_data']
            outputs = model(enhanced_data)
            predicted = (outputs > 0.5).float()  # 使用0.5作为阈值进行二值化
            print(f'Predicted: {predicted.cpu().numpy().flatten()}')

if __name__ == '__main__':
    train_model(epochs=40)
    # deploy_model()
