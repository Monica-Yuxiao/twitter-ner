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

a = torch.FloatTensor([1.0, 1.5]).view(1, -1)
a2 = torch.FloatTensor([5.0, 6.0]).view(1, -1)
print(helper.log_sum_exp(a))
print(helper.log_sum_exp(a2))


print(45 % 50)