import torch
from helper import argmax, log_sum_exp

import torch.nn as nn
from torch.autograd import Variable


'''
How a customized model is used in pytorch:
# compute model output
# pass inputs for a constructed model, it calls forward() internally and returns the prediction 
1. output_batch = model(train_batch)   

# calculate loss        
2. loss = loss_fn(output_batch, labels_batch)  

# clear previous gradients
3. optimizer.zero_grad()  

# compute gradients of all variables wrt loss
4. loss.backward()        

# perform updates using calculated gradients
5. optimizer.step()       
source: https://cs230-stanford.github.io/pytorch-getting-started.html
'''


class BLSTM_CRF1(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, debug=False):
        super(BLSTM_CRF1, self).__init__()
        self.debug = debug
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.hidden = self.init_hidden()

        '''
        construct embedding layer with parameters:
        vocab_size: size of dictionary of embedding (or num_embedding)
        embedding_dim: size of each embedding vector
        internally weight of shape (vocab_size, embedding_dim)
        '''
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        '''
        construct lstm layer with parameters:
        embedding_dim: # features in the input x (or input size)
        hidden_size = hidden_dim //2 for each direction, # features in the hidden state
        internally weight of shape (embedding_dim, hidden_dim // 2) from N(0, 1)
        '''
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        '''
        construct a linear layer with parameters
        a dense layer that multiple final feature vector with weight of shape (all_feats_dim, tagset_size)
        '''
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size - 2)

        # Matrix of transition parameters from N(0, 1).  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transition = nn.Parameter(torch.randn(self.tagset_size - 2, self.tagset_size - 1))
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        # self.transitions.data[tag_to_ix["<START>"], :] = -10000
        # self.transitions.data[:, tag_to_ix["<STOP>"]] = -10000
        # print("initiliazed transition:\n", self.transition)


    def init_hidden(self):
        '''
        initialize (h0, c0) as part of input for lstm
        :return:
        h0: initial hidden state for each instance in the batch,
            shape: (num_layer * num_directions, batch, hidden_size)
        c0: initial cell state for each instance in the batch,
            shape: (num_layers * num_directions, batch, hidden_size)
        where hidden_size is hidden_dim (specified) // 2 if bi-direction
        '''
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))


    def _get_lstm_hidden(self, sentence):
        # initiliaze (h0, c0)
        self.hidden = self.init_hidden()
        # sentence is a tensor vector size of len(sentence), input of embedding layer
        # output shape of word_embeds: (input*, embedding_size) where input* is len(sentence)
        # shape of embeds is (len(sentence), 1, embedding_size)
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        embeds = self.dropout(embeds)
        # lstm layer inputs: (input, (h0, c0))
        # 1st arg: input of shape(seq_len, batch, input_size)
        # 2nd atg: (h0, c0) initialized when construct the model, then updated
        # outputs: (output, (h_n, c_n))
        # output of shape (seq_len, batch, num_direction * hidden_size), hidden states at all time steps
        # (h_n, c_n): last time step hidden and cell, each of shape (num_layers*num_directions, batch, hidden_size)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # self.hidden being updated to continue for the next sentence
        # add a dropout layer
        lstm_out = self.dropout(lstm_out)
        return lstm_out.view(len(sentence), self.hidden_dim)


    def _get_emission(self, sentence):
        if self.debug: print("** inside get_emission **")
        lstm_out = self._get_lstm_hidden(sentence)
        # linear layer input: shape (len(sentence), all_feats_dim)
        # output of shape (len(sentence), tagset_size)
        # internally matmul input with weight of shape (all_feats_dim, tagset_size)
        emission = self.hidden2tag(lstm_out)
        if self.debug: print("shape of emission: ", emission.size())
        return emission


    def _forward_alg(self, emission):
        '''Do the forward algorithm to compute log of the partition function Z(x) := sum_over_all_y'(exp(score(x, y')))
        i.e. the denominator in P(y|x) = exp(score(x, y))/Z(x)
        where x is the sentence, y is the sequence of tags
        take neg-log likelihood: -log(P(y|x)) = log(Z(x)) - score(x, y)
        log(Z(x)) := log(sum_over_y'(exp(score(x, y'))) := log_sum_exp(score(x, y'))
        :return: alpha = log_sum_exp(score(x, y))
        '''

        if self.debug: print("** inside _forward_alg **")
        forward_var = self.transition[:, self.tag_to_ix["<START>"]] + emission[0].view(1, -1)
        if self.debug: print("forward score for the first word: ", forward_var)
        # Iterate through rest of the sentence
        for feat in emission[1:]:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size - 2):  # we only have O, B, I inside a sentence
                # broadcast the emission score: it is the same regardless of the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size - 2)
                # the score of transitioning to this next_tag from all previous tags (O, B ,I)
                trans_score = self.transition[next_tag, :self.tagset_size - 2].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the scores that lead to this tag.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))

            # now alphas_t is a list contains 3 scores for each tag state
            forward_var = torch.cat(alphas_t).view(1, -1)  # return shape [1, 3]
            if self.debug: print("forward score for the whole sentence: ", forward_var)

        # # now we have forward_var to the last word,
        # # add transition from O, B, I to STOP tag
        # terminal_var = forward_var + self.transition[self.tag_to_ix["<STOP>"], :self.tagset_size-2]
        alpha = log_sum_exp(forward_var)
        return alpha


    def _score_sentence(self, emission, tags):
        if self.debug: print("** inside _score_sentence **")
        # Gives the score of a provided tag sequence
        # emission of (len(sentence), tagset_size)
        score = Variable(torch.Tensor([0]))
        # append <START> to the true tags
        tags = torch.cat([torch.LongTensor([self.tag_to_ix["<START>"]]), tags])
        for i, feat in enumerate(emission):
            score = score + self.transition[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        # # add score of transition from last tag to STOP
        # score = score + self.transition[self.tag_to_ix["<STOP>"], tags[-1]]
        # print("** gold score: ", score)
        return score


    def neg_log_likelihood(self, sentence, true_tags):
        '''

        :param sentence(x): sentence of tensor size (len(sentence))
        :param true_tags(y): true tags of tensor size (len(sentence))
        :return: -log(P(y|x)) = -log {exp(score(x, y) / sum_exp(score(x, y')}
                             = -log(exp(score(x, y)) + log(sum_exp(score(x, y'))
                             = log_sum_exp(score(x, y') - score(x, y)
        '''
        if self.debug: print("** inside neg_log_likelihood **")
        emission = self._get_emission(sentence)
        forward_score = self._forward_alg(emission)
        if self.debug: print("forward score: ", forward_score)
        gold_score = self._score_sentence(emission, true_tags)
        if self.debug: print("gold score: ", gold_score)
        if self.debug: print("neg_log_likelihood: ", forward_score - gold_score)
        return forward_score - gold_score



    def _viterbi_decode(self, emission):
        '''
        :param emission: emission matrix of shape (len(sentence), tagset_size), output of _get_emission()
        :return: the most likely tag sequence for a give sentence(emission), and the score
        '''

        if self.debug: print("** inside _viterbi_decode **")

        forward_var = self.transition[:self.tagset_size - 2, self.tag_to_ix["<START>"]] + emission[0].view(1, -1)
        if self.debug: print("forward score for the first word: ", forward_var)

        backpointers = []

        for feat in emission[1:]:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size - 2):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transition[next_tag, :self.tagset_size - 2]  # shape of [1, 3]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                best_score = next_tag_var[0][best_tag_id].view(1)
                viterbivars_t.append(best_score)
            # Now add in the emission scores, and assign forward_var to the list
            # of 3 viterbi variables we just computed, forward_var again becomes shape of (1, 3)
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            if self.debug: print("forward score for current word: ", forward_var)
            backpointers.append(bptrs_t)

        # # Transition from the last tag (O,B,I) to STOP_TAG
        # print("forward score before add stop: ", forward_var)
        # terminal_var = forward_var + self.transition[self.tag_to_ix["<STOP>"], :self.tagset_size-2]
        best_tag_id = argmax(forward_var)
        path_score = forward_var[0][best_tag_id]
        if self.debug: print("final_viterbi_score: ", path_score)
        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        best_path.reverse()
        if self.debug: print("best path: ", best_path)
        return path_score, best_path


    def forward(self, sentence):
        if self.debug: print("** inside forward add_features**")
        # forward() gets called internally by output = model(input)
        # Get the emission scores from the BiLSTM
        emission = self._get_emission(sentence)
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(emission)
        return score, tag_seq
