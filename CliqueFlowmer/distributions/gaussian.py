import torch 
import math 


def log_likelihood(mu, sigma, x):
    ll = -((x-mu)**2)/(2 * sigma**2 + 1e-8) - torch.log(sigma + 1e-8) - math.log(2 * math.pi)/2 
    return ll.sum(-1)


def standard_kl(mu, sigma):
    kl = -torch.log(sigma + 1e-8) + (sigma**2 + mu**2)/2 - 0.5
    return kl.sum(-1)


def from_params(mu, log_sigma):
    sigma = torch.clamp(torch.exp(log_sigma), 1e-3, 1e2)
    sample = mu + sigma * torch.randn_like(mu)
    x = sample.detach()
    ll = log_likelihood(mu, sigma, x)
    return sample, ll 