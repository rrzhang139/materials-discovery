import torch 
import torch.nn as nn 
from torch.autograd import Variable

class Design(nn.Module):

    def __init__(self, param):

        super().__init__()
        self.param = Variable(param, requires_grad=True)
    
    @property
    def n_designs(self):
        return self.param.shape[0]
    
    def perturb(self, n, scale):
        noises = torch.randn((n,) + self.param.shape)
        perturbs = self.param[None, ...] + scale * noises
        return perturbs, noises
    
    def perturb_antithetic(self, n, scale):
        noises = torch.randn((n,) + self.param.shape).to(self.param.device)
        perturbs = self.param[None, ...] + scale * noises 
        anti_perturbs = self.param[None, ...] - scale * noises 
        return torch.stack([perturbs, anti_perturbs], dim=0), noises 