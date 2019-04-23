import torch

from ner.helper import *



# # a = torch.tensor([[1, 2, 3], [6,5,4]])
# a = torch.tensor([[1, 2, 3]])
# print(a.size())
# argmax = helper.argmax(a)
# b = helper.to_scalar(argmax)
# assert(b == 2)
#
#
# a = torch.randn((1, 3))
# print(a)
# print(log_sum_exp(a))
# b = helper.log_sum_exp(a)
# print(b)

a = torch.FloatTensor([0.9, 0.05, 0.05]).view(1, -1)
a2 = torch.FloatTensor([0.99, 0.05, 0.05]).view(1, -1)
print("log_sum_exp(a): ", log_sum_exp(a))
print("log_sum_exp(a2): ", log_sum_exp(a2))


