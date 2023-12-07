#%%
import torch

from components.timbre_transformer.TimberTransformer import TimbreTransformer

pt_file = "pt_file/train22_generator_best_31.pt"

model = TimbreTransformer(is_train=False, is_smooth=True, mlp_layer=6)
model.load_state_dict(torch.load(pt_file))
model.eval()
torch.save(model, "out_model_file.pt")
