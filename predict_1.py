from BLSTM_CRF1 import BLSTM_CRF1
from lstm import lstmTagger
from helper import load_data, prepare_sentence, prepare_tags
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

EMBEDDING_DIM = 16
HIDDEN_DIM = 16

model = BLSTM_CRF1(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load("out/lstm_crf_1/epoch80.hdf5"))

# predict on val set
true_tags = []
pred_tags = []
with torch.no_grad():
    for i in range(len(examples)):
        precheck_sent = prepare_sentence(examples[i][0], word_to_ix)
        precheck_tags = prepare_tags(examples[i][1], tag_to_ix)

        print("{} tags: {}\n".format(i, precheck_tags))
        predicts = model(precheck_sent)
        print("After training: {}\n ".format(predicts))

        true_tags.extend(precheck_tags.tolist())
        pred_tags.extend(predicts[1])


print(classification_report(true_tags, pred_tags))
