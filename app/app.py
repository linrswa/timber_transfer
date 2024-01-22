#%%
import torch
import random
import gradio as gr 
from glob import glob
from numpy import ndarray
from matplotlib import pyplot as plt

import sys
sys.path.append("..")
from data.dataset import NSynthDataset
from tools.utils import cal_loudness_norm
from components.timbre_transformer.TimberTransformer import TimbreTransformer

class GlobalInfo:
    def __init__(self):
        pt_dir = "../pt_file"
        self.current_pt_file_name = "train2_generator_best_0.pt"
        self.pt_file = f"{pt_dir}/{self.current_pt_file_name}"
        self.pt_file_list = sorted(glob(f"{pt_dir}/train*generator*.pt"))
        self.model = TimbreTransformer(is_train=False, is_smooth=True, mlp_layer=3)
        self.dataset = NSynthDataset(data_mode="train", sr=16000, frequency_with_confidence=True)
        self.model.eval()
        self.model.load_state_dict(torch.load(self.pt_file))

G_info = GlobalInfo()

def create_fig(data: ndarray) -> plt.Figure:
    fig = plt.figure()
    plt.plot(data)
    plt.close()
    return fig

def model_gen(s: ndarray, l_norm: ndarray, f:ndarray):
    def transfrom(np_array: ndarray) -> torch.Tensor:
        return torch.from_numpy(np_array).unsqueeze(0)
    s, l_norm, f = transfrom(s), transfrom(l_norm), transfrom(f)
    f = f[:, :-1, 0]
    _, _, rec_s, _, _, _ = G_info.model(s, l_norm, f)
    return rec_s

def sample_data():
    fn_with_path = random.choice(G_info.dataset.audio_list)
    fn = fn_with_path.split("/")[-1][:-4]
    _, s, l, f = G_info.dataset.getitem_by_fn(fn)
    return fn, s, l, f

def change_dataset(data_mode: str) -> str:
    G_info.dataset.set_data_mode(data_mode)
    return G_info.dataset.data_mode

def change_pt_file(pt_file: str):
    G_info.current_pt_file_name = pt_file.split("/")[-1]
    try:
        G_info.model.load_state_dict(torch.load(pt_file))
    except:
        raise gr.Error("load model failed")
    return G_info.current_pt_file_name

def generate_data():
    fn, s, l, f = sample_data()
    # create a matplotlib.figure.Figure for s
    fig_s = create_fig(s)
    rec_s = model_gen(s, cal_loudness_norm(l), f).squeeze().detach().numpy()
    fig_rec_s = create_fig(rec_s)
    
    return fn, (16000, s), fig_s, (16000, rec_s), fig_rec_s


with gr.Blocks() as app:
    with gr.Row():
        data_mode = gr.Radio(["train", "valid", "test"], label="Data Mode")

    with gr.Row():
        pt_file_selector = gr.Dropdown(G_info.pt_file_list, label="Model pt file") 
    
    with gr.Group():
        with gr.Row():
            signal_name = gr.Textbox(label="Dataset")
            pt_name = gr.Textbox(label="pt file", value=G_info.current_pt_file_name)

    with gr.Row():
        generate_button = gr.Button("generate data")
        generate_text = gr.Textbox(label="file name")

    with gr.Row():
        generate_image = gr.Plot(label="Ori signal")
        generate_audio = gr.Audio(label="Ori signal")

    with gr.Row():
        generate_rec_image = gr.Plot(label="Rec signal")
        generate_rec_audio = gr.Audio(label="Rec signal")

    data_mode.select(change_dataset, inputs=[data_mode], outputs=[signal_name])

    pt_file_selector.change(change_pt_file, inputs=[pt_file_selector], outputs=[pt_name])

    generate_button.click(
        generate_data,
        outputs=[
            generate_text,
            generate_audio,
            generate_image,
            generate_rec_audio,
            generate_rec_image
            ]
        )

app.launch()