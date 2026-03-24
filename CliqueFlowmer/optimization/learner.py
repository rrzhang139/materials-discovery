import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torch.autograd import Variable
import distributions.gaussian as gauss
import models.tools as tools 


class Learner(nn.Module):

    def __init__(self, design, model, **kwargs):

        super().__init__()
        self.design = design 
        self.model = model 
        self.model.eval()

        decay = kwargs.pop("decay")
        self.lr = kwargs.pop('lr') * math.sqrt(design.param.shape[-1])

        if 'sgd' in kwargs and kwargs.pop('sgd'):
            effective_decay = decay / self.lr
            self.optimizer = optim.SGD([self.design.param], lr=self.lr, weight_decay=effective_decay)

        else:
            self.optimizer = optim.AdamW([self.design.param], lr=self.lr, weight_decay=decay) 

        if 'structure_fn' in kwargs:
            self.structure_fn = kwargs.pop('structure_fn')
        else:
            self.structure_fn = (lambda x: x)

    def values(self):
        x = self.structure_fn(self.design.param)
        return self.model(x)

    def value(self):
        x = self.structure_fn(self.design.param)
        return self.model(x).mean()
    
    def train_step(self):
        pass 

    def design_fn(self):
        return self.design.param
    
    def best(self, n_best):
        values = self.values().view(-1)                    # shape: (N1,)
        best_values, best_indices = torch.topk(values, n_best, largest=False)
        return best_values.view(-1), best_indices.view(-1)



class GradientDescent(Learner):

    def __init__(self, design, model, **kwargs):
        super().__init__(design, model, **kwargs)

    def train_step(self):

        loss = self.value()
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        return {
            'loss': loss
        }



class ES(Learner):

    def __init__(self, design, model, **kwargs):
        super().__init__(design, model, **kwargs)
        self.n_pert = kwargs.pop('n_pert')
        self.scale_pert = kwargs.pop('scale_pert')
        self.antithetic = kwargs.pop('antithetic')
        self.rank = kwargs.pop('rank')

    def train_step(self):
        
        if self.antithetic:
            population, noises = self.design.perturb_antithetic(self.n_pert, self.scale_pert)
            population_struct = self.structure_fn(population)
            shape = population_struct.shape
            reshaped = population_struct.reshape(-1, shape[-2], shape[-1])
            
            vals = self.model(reshaped)
            vals = vals.reshape(2, self.n_pert, -1)

            if self.rank:
                vals = torch.cat([vals[0], vals[1]], dim=0)
                vals = tools.rank(vals, dim=0)
                vals = tools.standardize(vals, dim=0)
                vals = torch.stack(torch.chunk(vals, 2, 0), dim=0)

            vals = (vals[0] - vals[1])/2
        
        else:
            population, noises = self.design.perturb(self.n_pert, self.scale_pert)
            population_struct = self.structure_fn(population)
            shape = population_struct.shape
            reshaped = population_struct.reshape(-1, shape[-2], shape[-1])
            
            vals = self.model(reshaped)
            vals = vals.reshape(self.n_pert, -1)
            
            if self.rank:
                vals = tools.rank(vals, dim=0)
                vals = tools.standardize(vals, dim=0)
        
        es_grad = vals[..., None] * noises / self.scale_pert
        es_grad = es_grad.mean(0)

        self.design.param.grad = es_grad.clone()
        self.optimizer.step()
        self.optimizer.zero_grad()
