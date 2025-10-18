import torch
from safetensors.torch import save_file

state_dict = torch.load("FLUX_lineart/flux_controlnet/diffusion_pytorch_model.bin", map_location="cpu")
save_file(state_dict, "FLUX_lineart/flux_controlnet/diffusion_pytorch_model.safetensors")
