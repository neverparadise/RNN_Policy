import torch

torch.manual_seed(0)
print(torch.randn([1]))
torch.manual_seed(0)
print(torch.randn([1]))
