import torch

from helper import *



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

b = [3]
a = [0] * 5
b.extend(a)
print(b)

print(7//3)


a = torch.tensor([[-0.1001, -3.3287, -2.8209],[-0.1066, -3.0062, -2.9638],[-0.1084,-3.0056,-2.9338]])
print(a)

to_labels(a)