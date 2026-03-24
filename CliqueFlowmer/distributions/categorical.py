import torch  
from torch.distributions import categorical


def log_likelihood(probs, x):
    #
    # Assume probs are (N, K, C) and x is (N, K, 1)
    #
    the_probs = torch.gather(probs, 2, x).squeeze(-1)
    return torch.log(the_probs + 1e-8).sum(-1)


def from_params(probs):
    dist = categorical.Categorical(probs)
    sample = dist.sample()
    log_lik = dist.log_prob(sample)
    return sample, log_lik