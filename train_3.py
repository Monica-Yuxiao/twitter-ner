from helper import *
import torch
from CNN_BLSTM_CRF import CNN_BLSTM_CRF
import torch.optim as optim
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pickle
import numpy as np


torch.manual_seed(1)

train_path = "data/twitter_ner/train.txt"

# load training data
examples = load_data(train_path)
print("how many examples: ", len(examples))
print("first example:\n", examples[0])
print("last example:\n", examples[-1])

features = add_features(examples)
print("feature for 1st example: \n", features[0])
print("feature for last example: \n", features[-1])

tag_to_ix = {"B": 2, "I": 1, "O": 0, "<START>": 3, "<STOP>": 4}

vocab = dict()
for sentence, tags in examples:
    for word in sentence:
        if word not in vocab:
            vocab[word] = len(vocab)

print("== vocab size: ", len(vocab))

word_to_ix = dict()
word_to_ix["<UNK>"] = 0

word_count = {}
for sentence, tags in examples:
    for word in sentence:
        if word in word_count:
            word_count[word] += 1
        else: word_count[word] = 1

rare_words = []
freq = 1
for k, v in word_count.items():
    if v <= freq:
        rare_words.append(k)
    else:
        word_to_ix[k] = len(word_to_ix)

print("== how many rare word: ", len(rare_words))
print("== word_to_ix size: ", len(word_to_ix))


out = open("out/char_word_to_ix", "wb")
pickle.dump(word_to_ix, out)
out.close()

out = open("out/char_tag_to_ix", "wb")
pickle.dump(tag_to_ix, out)
out.close()


char_to_ix = dict()
char_to_ix["<UNK>"] = 0
for sentence, _ in examples:
    for word in sentence:
        for char in word:
            if char not in char_to_ix:
                char_to_ix[char] = len(char_to_ix)

print("== char size: ", len(char_to_ix))
print(char_to_ix)

out = open("out/char_to_ix", "wb")
pickle.dump(char_to_ix, out)
out.close()

# get lexicon features dimension from length of 1st word's vector in 1st sentence
add_features_dim = len(features[0][0])
print("how many additional features: ", add_features_dim)

EMBEDDING_DIM = 16
HIDDEN_DIM = 16
MAX_CHARS = 10

print("len(char_to_ix): ", len(char_to_ix))
print("len(word_to_ix): ", len(word_to_ix))
print("features_dim: ", add_features_dim)

model = CNN_BLSTM_CRF(len(char_to_ix), MAX_CHARS, len(word_to_ix), tag_to_ix, add_features_dim, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# hold out some for validation
train = []
val = []
features_train = []
features_val = []
for i in range(len(examples)):
    rng = np.random.random_sample()
    if rng > 0.8:
        val.append(examples[i])
        features_val.append(features[i])
    else:
        train.append(examples[i])
        features_train.append((features[i]))

print("{} training instances, {} validation instances".format(len(train), len(val)))
# calculate validation loss
def calc_val_loss(val):
    loss = []
    for j in range(len(val)):
        sentence = val[j][0]
        tags = val[j][1]
        sentence_in = prepare_sentence(sentence, word_to_ix)
        targets = prepare_tags(tags, tag_to_ix)
        features_in = prepare_features(features_val[j])
        chars_in = prepare_char_level_sentence(sentence, char_to_ix, MAX_CHARS)
        nll = model.neg_log_likelihood(sentence_in, targets, features_in, chars_in)
        loss.append(nll.item())
    return sum(loss) / len(loss)


NUM_TO_TRAIN = len(train)
NUM_TO_PRED = 20
EPOCHES = 150

# start training
train_loss = []
# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(EPOCHES):  # again, normally you would NOT do 300 epochs, it is toy data
    print("epoch %d" % epoch)
    # generate new order of index
    idx = np.random.choice(NUM_TO_TRAIN, replace=False, size=NUM_TO_TRAIN)
    for i in idx:
        sentence = train[i][0]
        tags = train[i][1]
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sentence(sentence, word_to_ix)
        targets = prepare_tags(tags, tag_to_ix)
        # print("how many tags in this sentence: ", targets.size())
        features_in = prepare_features(features_train[i])
        chars_in = prepare_char_level_sentence(sentence, char_to_ix, MAX_CHARS)
        # print("chars_in size: ", chars_in.size())

        # Step 3. Run our forward pass.
        predicts = model(sentence_in, features_in, chars_in)
        loss = model.neg_log_likelihood(sentence_in, targets, features_in, chars_in)

        # if i % 10 == 0: print("loss: ", loss)
        train_loss.append(loss.item())

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

    # print average loss after each iteration
    val_loss = calc_val_loss(val)
    print("epoch {}: train_loss = {:4f} val_loss = {:4f}".format(epoch, sum(train_loss) / len(train_loss), val_loss))

    if epoch % 20 == 0:
        torch.save(model.state_dict(), "out/lstm_crf_char/epoch{}.hdf5".format(epoch))


# Check a few sentence predictions after training
with torch.no_grad():
    true_tags = []
    pred_tags = []
    for i in range(NUM_TO_PRED):
        precheck_sent = prepare_sentence(train[i][0], word_to_ix)
        precheck_tags = prepare_tags(train[i][1], tag_to_ix)
        precheck_features = prepare_features(features_train[i])
        # print("{} sent: {}\n".format(i, precheck_sent))
        print("{} tags: {}\n".format(i, precheck_tags))
        precheck_chars = prepare_char_level_sentence(train[i][0], char_to_ix, MAX_CHARS)
        predicts = model(precheck_sent, precheck_features, precheck_chars)
        print("After training: {}\n ".format(predicts))

        true_tags.extend(precheck_tags.tolist())
        pred_tags.extend(predicts[1])

    print("NUM_TO_TRAIN: {} | EPOCHES: {}".format(NUM_TO_TRAIN, EPOCHES))
    print(classification_report(true_tags, pred_tags))