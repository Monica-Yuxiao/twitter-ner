import torch


# def to_scalar(var):
#     # convert tensor of 1 element into a scalar
#     assert len(var.view(-1).data.tolist()) == 1, \
#         print("Can't convert to scalar. \nInput: {}".format(var))
#     return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec): # vec in shape of (1, x)
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


# def log_sum_exp(vec):
#     return torch.log(torch.sum(torch.exp(vec)))


