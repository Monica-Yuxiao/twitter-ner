from ner.helper import start_with_capital, contains_special
from ner.helper import prepare_sentence, prepare_features, prepare_tags
import torch
from ner.model import BiLSTM_CRF
import torch.optim as optim
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


torch.manual_seed(1)

EMBEDDING_DIM = 5
HIDDEN_DIM = 4

def load_data(infpath):
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


def add_features():
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


val_path = "../data/twitter_ner/validation.txt"

# load training data
examples = []
examples = load_data(val_path)
print("first example:\n", examples[0])
print("last example:\n", examples[-1])

features = add_features()
print("feature for 1st example: \n", features[0])
print("feature for last example: \n", features[-1])

tag_to_ix = {"B": 2, "I": 1, "O": 0, "<START>": 3, "<STOP>": 4}
word_to_ix = {}
for sentence, tags in examples:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

# print("== word_to_ix: \n", word_to_ix)
print("== word_to_ix size: ", len(word_to_ix))

# get lexicon features dimension from length of 1st word's vector in 1st sentence
features_dim = len(features[0][0])
print("how many additional features: ", features_dim)
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, features_dim, EMBEDDING_DIM, HIDDEN_DIM, False)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

NUM_TO_TRAIN = 100
NUM_TO_PRED = 3
EPOCHES = 100

# Check a few sentence predictions before training
with torch.no_grad():
    true_tags = []
    pred_tags = []
    for i in range(NUM_TO_PRED):
        precheck_sent = prepare_sentence(examples[i][0], word_to_ix)
        precheck_tags = prepare_tags(examples[i][1], tag_to_ix)
        precheck_features = prepare_features(features[i])
        # print("{} sent: {}\n".format(i, precheck_sent))
        # print("{} tags: {}\n".format(i, precheck_tags))
        predicts = model(precheck_sent, precheck_features)
        # print("Before training: {}\n ".format(predicts))

        true_tags.extend(precheck_tags.tolist())
        pred_tags.extend(predicts[1])
    # print(true_tags)
    # print(pred_tags)
    print(classification_report(true_tags, pred_tags))


# start training
neg_log_likelihoods = []
# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(EPOCHES):  # again, normally you would NOT do 300 epochs, it is toy data
    print("epoch %d" % epoch)
    for i in range(NUM_TO_TRAIN):
        sentence = examples[i][0]
        tags = examples[i][1]
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sentence(sentence, word_to_ix)
        targets = prepare_tags(tags, tag_to_ix)
        features_in = prepare_features(features[i])
        # print("more features for current sentence size: ", features_in.size())

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets, features_in)
        # if i % 10 == 0: print("loss: ", loss)
        neg_log_likelihoods.append(loss)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check a few sentence predictions after training
with torch.no_grad():
    true_tags = []
    pred_tags = []
    for i in range(3):
        precheck_sent = prepare_sentence(examples[i][0], word_to_ix)
        precheck_tags = prepare_tags(examples[i][1], tag_to_ix)
        precheck_features = prepare_features(features[i])
        # print("{} sent: {}\n".format(i, precheck_sent))
        print("{} tags: {}\n".format(i, precheck_tags))
        predicts = model(precheck_sent, precheck_features)
        print("After training: {}\n ".format(predicts))

        true_tags.extend(precheck_tags.tolist())
        pred_tags.extend(predicts[1])
    # print(true_tags)
    # print(pred_tags)
    print(classification_report(true_tags, pred_tags))



plt.plot(neg_log_likelihoods)
plt.show()