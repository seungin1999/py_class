import torch
import torch.nn as nn

class TemperatureScaling(nn.Module):
    def __init__(self, model, initial_temp=0.1):
        super(TemperatureScaling, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * initial_temp)
    
    def forward(self, inputs):
        logits = self.model(inputs)
        return logits / self.temperature