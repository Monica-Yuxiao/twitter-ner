import torch
import re


def prepare_sentence(seq, to_ix):
    idxs = [to_ix[w] if w in to_ix else to_ix["<UNK>"] for w in seq]
    return torch.LongTensor(idxs)

def prepare_tags(seq, to_ix):
    # B-person maps to B
    idxs = [to_ix[w[0]] for w in seq]
    return torch.LongTensor(idxs)


def prepare_features(feature_sequence):
    return torch.FloatTensor(feature_sequence)


def load_data(infpath):
    examples = []
    new_example = ([], [])
    with open(infpath) as inf:
        line = inf.readline()
        while line:
            # print(">", line.strip(), "<")
            if line.strip() != "":  # add to tuple
                word, tag = line.strip().split()
                new_example[0].append(word)
                new_example[1].append(tag)
            else:  # add current example to list, then start a new example
                examples.append(new_example)
                new_example = ([], [])
            line = inf.readline()
        examples.append(new_example)
    return examples


def add_features(examples):
    features = list()
    for example in examples:
        sentence_feature = list()
        for word in example[0]:
            word_feature = list()
            word_feature.append(start_with_capital(word))
            word_feature.append(contains_special(word))
            # word_feature.append(no_alphabet(word))
            sentence_feature.append(word_feature)
        features.append(sentence_feature)
    return features


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec): # vec in shape of (1, x)
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))



def start_with_capital(word):
    return 1 if re.match(r'[A-Z]', word[0]) else 0


def contains_special(word):
    return 1 if re.search(r'[@_!#$%^&*()<>?/\|}{~:]', word) else 0


def contains_alphabet(word):
    return 1 if re.search(r'A-Z', word) else 0

