import torch
import torch.nn as nn
import torch.nn.functional as F
from model import resnet

class HTSModule(nn.Module) :
    def __init__(self, num_classes):
        super(HTSModule, self).__init__()
        self.wL = nn.Parameter(torch.tensor(1.0))
        self.wH = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.num_classes = num_classes
        
    def forward(self, logits):
        probs = F.softmax(logits, dim=-1)
        
        entropy = -torch.sum(probs * torch.log(probs+ 1e-8), dim=-1)
        
        normalizaed_entropy = entropy / torch.log(torch.tensor(self.num_classes, dtype = torch.float32))
        
        temperature = F.softplus(self.wL * logits + self.wH * torch.log(normalizaed_entropy.unsqueeze(-1)) + self.b)
        
        calibrated_logits = logits / temperature
        
        return calibrated_logits
        
class ResNetWithHTS(nn.Module) :
    def __init__(self, num_class, **kwargs):
        super(ResNetWithHTS, self).__init__()      
        self.resnet = resnet.resnet110(**kwargs)
        self.hts = HTSModule(num_class)  
        
    def forward(self, x):
        logits = self.resnet(x)
        
        calibrated_logits = self.hts(logits)
        
        return calibrated_logits


# class HTSModule(nn.Module):
#     def __init__(self, num_classes):
#         super(HTSModule, self).__init__()
#         # 파라미터 초기값 조정 (작은 값으로 초기화)
#         self.wH = nn.Parameter(torch.tensor(0.1))  # Log 엔트로피 항목 가중치
#         self.b = nn.Parameter(torch.tensor(0.0))  # Bias
#         self.num_classes = num_classes  # 클래스 개수

#     def forward(self, logits):
#         # 소프트맥스 확률 계산
#         probs = F.softmax(logits, dim=-1)
        
#         # 엔트로피 계산
#         entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        
#         # 엔트로피 정규화 (클래스 개수에 따라)
#         normalized_entropy = entropy / torch.log(torch.tensor(self.num_classes, dtype=torch.float32, device=logits.device))
        
#         # 안정화된 엔트로피 계산 (clamp로 최소값 설정)
#         safe_entropy = torch.clamp(normalized_entropy, min=1e-8)
        
#         # 온도 계산 (Log 엔트로피 항목만 사용)
#         temperature = F.softplus(self.wH * torch.log(safe_entropy) + self.b)
        
#         # 온도로 조정된 로짓 계산
#         calibrated_logits = logits / temperature.unsqueeze(-1)
        
#         return calibrated_logits

# class ResNetWithHTS(nn.Module) :
#     def __init__(self, num_class, **kwargs):
#         super(ResNetWithHTS, self).__init__()      
#         self.resnet = resnet.resnet110(**kwargs)
#         self.hts = HTSModule(num_class)  
        
#     def forward(self, x):
#         logits = self.resnet(x)
        
#         calibrated_logits = self.hts(logits)
        
#         return logits, calibrated_logits