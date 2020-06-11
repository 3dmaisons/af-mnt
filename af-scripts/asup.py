import torch
B, T = 2, 3 # 128, 63
data = torch.rand((B, T))
# mask = torch.empty(B, T).random_(2).bool()
mask = data.gt(0.5) * torch.empty(B, T).random_(2).bool()

print(data)
tmp = data.detach()
print(type(tmp), tmp)
tmp = data.detach().sum().item()
print(type(tmp), tmp)
print(data.detach().item())

p1 = data * mask
p2 = mask * data
p3 = data * mask.int().float()
print((p1==p2).all())
print((p1==p3).all())