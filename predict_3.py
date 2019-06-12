from CNN_BLSTM_CRF import CNN_BLSTM_CRF
from helper import load_data, add_features, prepare_sentence, prepare_tags, prepare_features, prepare_char_level_sentence
import pickle
import torch
from sklearn.metrics import classification_report


inf = open("out/char_word_to_ix", "rb")
word_to_ix = pickle.load(inf)
inf.close()


with open("out/char_tag_to_ix", "rb") as inf:
    tag_to_ix = pickle.load(inf)

with open("out/char_to_ix", "rb") as inf:
    char_to_ix = pickle.load(inf)


#load val set
val_path = "data/twitter_ner/validation.txt"
examples = load_data(val_path)
features = add_features(examples)

features_dim = len(features[0][0])
print("how many additional features: ", features_dim)
EMBEDDING_DIM = 16
HIDDEN_DIM = 16
MAX_CHARS = 10

print("len(char_to_ix): ", len(char_to_ix))
print("len(word_to_ix): ", len(word_to_ix))
print("features_dim: ", features_dim)
model = CNN_BLSTM_CRF(len(char_to_ix), MAX_CHARS, len(word_to_ix), tag_to_ix, features_dim, EMBEDDING_DIM, HIDDEN_DIM, False)
model.load_state_dict(torch.load("out/lstm_crf_char/epoch20.hdf5"))

# predict on val set
true_tags = []
pred_tags = []
with torch.no_grad():
    for i in range(len(examples)):
        precheck_sent = prepare_sentence(examples[i][0], word_to_ix)
        precheck_tags = prepare_tags(examples[i][1], tag_to_ix)
        precheck_features = prepare_features(features[i])
        precheck_chars = prepare_char_level_sentence(examples[i][0], char_to_ix, MAX_CHARS)
        predicts = model(precheck_sent, precheck_features, precheck_chars)
        print("After training: {}\n ".format(predicts))

        true_tags.extend(precheck_tags.tolist())
        pred_tags.extend(predicts[1])


print(classification_report(true_tags, pred_tags))