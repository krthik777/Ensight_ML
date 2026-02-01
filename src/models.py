# src/models.py
import torch
import torch.nn as nn

class MultiOutputLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=9, dropout=0.3):
        super(MultiOutputLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class ConvLSTM(nn.Module):
    def __init__(self, input_channels=1, output_size=9, seq_len=100):
        super(ConvLSTM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(64, 128, 2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def get_model(model_name, input_size=1, output_size=9, **kwargs):
    models = {
        'lstm': MultiOutputLSTM,
        'convlstm': ConvLSTM
    }
    
    if model_name == 'convlstm':
        return ConvLSTM(input_channels=input_size, output_size=output_size, **kwargs)
    else:
        return MultiOutputLSTM(input_size=input_size, output_size=output_size, **kwargs)