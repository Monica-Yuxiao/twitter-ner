import torch

from ner import helper



# # a = torch.tensor([[1, 2, 3], [6,5,4]])
# a = torch.tensor([[1, 2, 3]])
# print(a.size())
# argmax = helper.argmax(a)
# b = helper.to_scalar(argmax)
# assert(b == 2)
#
#
a = torch.randn((1, 3))
print(a)
# b = helper.log_sum_exp(a)
# print(b)

b = helper.log_sum_exp_0(a)
print(b)
