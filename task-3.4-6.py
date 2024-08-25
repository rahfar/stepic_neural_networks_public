import torch

mse_loss = torch.nn.MSELoss()
ce_loss = torch.nn.CrossEntropyLoss()

y_hat = torch.tensor([1, 0.3, 1], dtype=torch.float32)
y = torch.tensor([1, 0, 0], dtype=torch.float32)

print(mse_loss(y_hat, y))
print(ce_loss(y_hat, y))
