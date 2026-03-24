import torch 


def chain_of_cliques(n_cliques=1, dim=3, overlap=0, permute=False):

    assert dim > overlap

    n = (dim - overlap) * (n_cliques - 1) + dim
    perm = torch.randperm(n) if permute else torch.arange(n)
    cliques = [
        perm[i * (dim - overlap) : i * (dim - overlap) + dim] for i in range(n_cliques)
    ]

    return torch.stack(cliques, dim=0)


def separate_latents(x, index_matrix):

    n_cliques = index_matrix.shape[0]
    n_dims = len(x.shape)

    #
    # Copy x n_cliques times before the penultimate dimension
    #
    x = x.view(x.shape[:-1] + (1,) + x.shape[-1:])
    x = x.repeat((n_dims - 1) * (1,) + (n_cliques,) + (1,))

    #
    # Align the dimensions of the index matrix with x
    #
    m = index_matrix.view((n_dims - 1) * (1,) + index_matrix.shape)
    m = m.repeat(x.shape[:-2] + 2 * (1,))
    
    return torch.gather(x, -1, m)