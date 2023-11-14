#%%
import torch

from components.ddsp_modify.ddsp import DDSP

pt_file = "pt_file/train20_generator_best_6.pt"

model = DDSP(is_train=False, is_smooth=True, mlp_layer=6)
model.load_state_dict(torch.load(pt_file))
model.eval()
torch.save(model, "out_model_file.pt")
