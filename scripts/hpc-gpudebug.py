import torch

print(f"Is CUDA available?: {torch.cuda.is_available()}")
print(f"Number of CUDA devices: {torch.cuda.device_count()}")

device = torch.device('cuda')
print(f"A torch tensor: {torch.rand(5).to(device)}")