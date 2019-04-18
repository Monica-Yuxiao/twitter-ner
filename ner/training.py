from ner.features import add_features
from ner.model import BiLSTM_CRF
import torch
import torch.optim as optim
import matplotlib.pyplot as plt


torch.manual_seed(1)

EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
training_data = [(
    "The Wall Street Journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "Georgia Tech is a university in Georgia".split(),
    "B I O O O O B".split()
)]
print("== training data ==\n", training_data)


more_features = add_features(training_data)
print("Features:\n", more_features)


tag_to_ix = {"B": 0, "I": 1, "O": 2, "<START>": 3, "<STOP>": 4}
word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

print("== word_to_ix: \n", word_to_ix)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.LongTensor(idxs)


def prepare_features(feature_sequence):
    return torch.LongTensor(feature_sequence)


# for i in range(len(training_data)):
#     sentence = training_data[i][0]
#     tags = training_data[i][1]
#     features = more_features[i]
#     sentence_in = prepare_sequence(sentence, word_to_ix)
#     features_in = prepare_features(features)
#     print("sentence_in: ", sentence_in)
#     print("features_in: ", features_in)
#     print("sentence size: ", sentence_in.size())
#     print("feature size: ", features_in.size())

features_dim = len(more_features[0][0])
print("how many additional features: ", features_dim)
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, features_dim, EMBEDDING_DIM, HIDDEN_DIM, False)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.LongTensor([tag_to_ix[t] for t in training_data[0][1]])
    precheck_features = torch.FloatTensor(more_features[0])
    print("precheck_sent: ", precheck_sent)
    print("precheck_tags: ", precheck_tags)
    print("Before training: \n ", model(precheck_sent, precheck_features))


neg_log_likelihoods = []
# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for i in range(len(training_data)):
    # for sentence, tags in training_data:
        sentence = training_data[i][0]
        tags = training_data[i][1]
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.LongTensor([tag_to_ix[t] for t in tags])
        features_in = torch.FloatTensor(more_features[i])
        # print("more features for current sentence size: ", features_in.size())

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets, features_in)
        neg_log_likelihoods.append(loss)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print("After training 1st:\n", model(precheck_sent, torch.FloatTensor(more_features[0])))
    precheck_sent = prepare_sequence(training_data[1][0], word_to_ix)
    print("After training 2nd:\n", model(precheck_sent, torch.FloatTensor(more_features[1])))


plt.plot(neg_log_likelihoods)
plt.show()