import torch
import torch.nn as nn
import torch.optim as optim

class RegressionNN(nn.Module):
    def __init__(self, input_size=6, output_size=19):
        super(RegressionNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),

            nn.Linear(64, 128),
            nn.ReLU(),


            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),
            

            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        return self.model(x)
    




class MultiHeadRegressor(nn.Module):
    def __init__(self, input_size=6, output_size=19, dropout_rate=0.2):
        super().__init__()
        # Shared base
        self.shared = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout_rate),

            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            # nn.Linear(256, 128),
            # nn.ReLU(),

            # nn.Linear(128, 64),
            # nn.ReLU(),
        )

        # Four heads
        self.head1 = nn.Linear(64, 6)
        self.head2 = nn.Linear(64, 6)
        self.head3 = nn.Linear(64, 6)
        self.head4 = nn.Linear(64, 1)

    def forward(self, x):
        base = self.shared(x)
        out1 = self.head1(base)
        out2 = self.head2(base)
        out3 = self.head3(base)
        out4 = self.head4(base)
        return torch.cat([out1, out2, out3, out4], dim=1)