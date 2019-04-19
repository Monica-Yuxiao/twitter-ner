import torch
import torch.nn as nn


hidden_dim = 5
a = torch.randn(2, 1, hidden_dim // 2)
print("a:\n", a)
print("a.size:\n", a.size())


def init_hidden(hidden_dim):
    return (torch.randn(2, 1, hidden_dim // 2),
            torch.randn(2, 1, hidden_dim // 2))


print("init_hidden(hidden_dim):\n", init_hidden(hidden_dim))

transition = nn.Parameter(torch.randn(1, 3))
print(transition)

m = nn.Linear(5, 2)
input = torch.randn(10, 5)
output = m(input)
print("linear output size: ", output.size())
print("linear output: \n", output)
emission = output
print("emission: ", emission)
print("transition: ", transition)
print("emission[0]: ", emission[0])
# score = transition + emission[0].view(1, -1)
# print("sum: ", score)
# print("sum shape: ", score.size())


