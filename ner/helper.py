import torch
import re



def start_with_capital(word):
    return 1 if re.match(r'[A-Z]', word[0]) else 0


def contains_special(word):
    return 1 if re.search(r'[@_!#$%^&*()<>?/\|}{~:]', word) else 0


def prepare_sentence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.LongTensor(idxs)



def prepare_tags(seq, to_ix):
    # B-person maps to B
    idxs = [to_ix[w[0]] for w in seq]
    return torch.LongTensor(idxs)


def prepare_features(feature_sequence):
    return torch.FloatTensor(feature_sequence)


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec): # vec in shape of (1, x)
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))



