import torch
import torch.nn as nn
import numpy as np


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

# test cnn
m = nn.Sequential(nn.Conv1d(in_channels=93, out_channels=1, kernel_size=3, padding=2),
                                   nn.ReLU(),
                                nn.MaxPool1d(3))
# (N = batch_size = # words in a sentence, C_in = char_dim, L_in = # chars in a word
input = torch.randn(27, 93, 5)
output = m(input)
output = output.view(27, -1)
print("output size: ", output.size())
print(output)

for i in range(10):
    rng = np.random.random_sample()
    print(rng)



a = 10
b = np.arange(a)
print("b: ", b)
idx = np.random.choice(10, replace=False, size=10)
print(idx)
