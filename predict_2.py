from BLSTM_CRF2 import BLSTM_CRF2
from helper import load_data, add_features, prepare_sentence, prepare_tags, prepare_features
import pickle
import torch
from sklearn.metrics import classification_report


inf = open("out/word_to_ix", "rb")
word_to_ix = pickle.load(inf)
inf.close()


with open("out/tag_to_ix", "rb") as inf:
    tag_to_ix = pickle.load(inf)



#load val set
val_path = "data/twitter_ner/validation.txt"
examples = load_data(val_path)
features = add_features(examples)


features_dim = len(features[0][0])
print("feature dim: ", features_dim)
EMBEDDING_DIM = 16
HIDDEN_DIM = 16

model = BLSTM_CRF2(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, features_dim)
model.load_state_dict(torch.load("out/lstm_crf_2/epoch20.hdf5"))

# predict on val set
true_tags = []
pred_tags = []
with torch.no_grad():
    for i in range(len(examples)):
        precheck_sent = prepare_sentence(examples[i][0], word_to_ix)
        precheck_tags = prepare_tags(examples[i][1], tag_to_ix)
        precheck_features = prepare_features(features[i])
        print("{} tags: {}\n".format(i, precheck_tags))
        predicts = model(precheck_sent, precheck_features)
        print("After training: {}\n ".format(predicts))

        true_tags.extend(precheck_tags.tolist())
        pred_tags.extend(predicts[1])


print(classification_report(true_tags, pred_tags))
