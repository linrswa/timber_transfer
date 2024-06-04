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


def create_fig(data: ndarray) -> plt.Figure:
    fig = plt.figure()
    plt.plot(data)
    plt.close()
    return fig

class GlobalInfo:
    def __init__(self):
        pt_dir = "../pt_file"
        run_name = "decoder_v6_1"
        self.current_pt_file_name = f"{run_name}_generator_best_0.pt"
        self.pt_file = f"{pt_dir}/{self.current_pt_file_name}"
        self.pt_file_list = sorted(glob(f"{pt_dir}/{run_name}*.pt"))
        self.model = TimbreTransformer(is_train=False, is_smooth=True, timbre_emb_dim=256)
        self.dataset = NSynthDataset(data_mode="train", sr=16000, frequency_with_confidence=True)
        self.source_audio_file_name = None
        self.target_audio_file_name = None
        self.model_input_selection = ("source", "source", "source")
        self.model.eval()
        self.model.load_state_dict(torch.load(self.pt_file))


    def sample_data(self, t: str = "source"):
        fn_with_path = random.choice(self.dataset.audio_list)
        fn = fn_with_path.split("/")[-1][:-4]
        fn, s, _, _ = self.dataset.getitem_by_filename(fn)
        if t == "source":
            self.source_audio_file_name = fn
        else :
            self.target_audio_file_name = fn
        fig_s = create_fig(s)
        return fn, (16000, s), fig_s
        
    def sampel_source_audio_data(self):
        return self.sample_data("source")

    def sampel_target_audio_data(self):
        return self.sample_data("target")

    def generate_model_input(self):
        source_fn = self.source_audio_file_name
        target_fn = self.target_audio_file_name
        _, source_s, source_l, source_f = self.dataset.getitem_by_filename(source_fn)
        _, target_s, target_l, target_f = self.dataset.getitem_by_filename(target_fn)
        s = target_s if self.model_input_selection[0] == "target" else source_s
        l = target_l if self.model_input_selection[1] == "target" else source_l
        f = target_f if self.model_input_selection[2] == "target" else source_f
        return s, l, f
        
    def generate_output(self):
        s, l, f = self.generate_model_input()
        rec_s = self.model_gen(s, cal_loudness_norm(l), f).squeeze().detach().numpy()
        fig_rec_s = create_fig(rec_s)
        return (16000, rec_s), fig_rec_s

    def model_gen(self, s: ndarray, l_norm: ndarray, f:ndarray):
        transfrom = lambda x_array: torch.from_numpy(x_array).unsqueeze(0)
        s, l_norm, f = transfrom(s), transfrom(l_norm), transfrom(f)
        f = f[:, :-1, 0]
        _, _, rec_s, _, _, _ = self.model(s, l_norm, f)
        return rec_s
    
    def change_dataset(self, data_mode: str) -> str:
        self.dataset.set_data_mode(data_mode)
        return self.dataset.data_mode

    def change_pt_file(self, pt_file: str):
        self.current_pt_file_name = pt_file.split("/")[-1]
        try:
            self.model.load_state_dict(torch.load(pt_file))
        except:
            raise gr.Error("load model failed")
        return self.current_pt_file_name
    
    def change_model_input(self, timbre:str , loudness:str, f0:str):
        selection = [timbre, loudness, f0]
        for i, item in enumerate(selection):
            if item == None:
                selection[i] = "source"
        self.model_input_selection = selection
        return f"T: {selection[0]}, L: {selection[1]}, F: {selection[2]}"


G = GlobalInfo()


with gr.Blocks() as app:
    with gr.Row():
        data_mode_selector = gr.Radio(["train", "valid", "test"], label="Data Mode")

    with gr.Row():
        pt_file_selector = gr.Dropdown(G.pt_file_list, label="Model pt file") 
    
    with gr.Group():
        with gr.Row():
            signal_name = gr.Textbox(label="Dataset")
            pt_name = gr.Textbox(label="pt file", value=G.current_pt_file_name)

    with gr.Column():
        source_text = gr.Textbox(
            label="source file",
            value="T: source, L: source, F: source"
            )
        source_sample_button = gr.Button("source sample")
        with gr.Row():
            source_image = gr.Plot(label="Source signal")
            source_audio = gr.Audio(label="Source signal")

    with gr.Column():
        target_text = gr.Textbox(label="target file", value=G.target_audio_file_name)
        target_sample_button = gr.Button("target sample")
        with gr.Row():
            target_image = gr.Plot(label="Target signal")
            target_audio = gr.Audio(label="Target signal")
    
    with gr.Column():
        with gr.Row():
            timbre_selector = gr.Radio(["source", "target"], label="Timbre")
            loudness_selector = gr.Radio(["source", "target"], label="Loudness")
            f0_selector = gr.Radio(["source", "target"], label="F0")

    with gr.Column():
        generate_selection_text = gr.Textbox(label="generate selection", value=G.model_input_selection)
        generate_button = gr.Button("generate data")
        with gr.Row():
            rec_image = gr.Plot(label="Rec signal")
            rec_audio = gr.Audio(label="Rec signal")

    data_mode_selector.select(G.change_dataset, inputs=[data_mode_selector], outputs=[signal_name])
    pt_file_selector.change(G.change_pt_file, inputs=[pt_file_selector], outputs=[pt_name])
    timbre_selector.change(
        G.change_model_input,
        inputs=[timbre_selector, loudness_selector, f0_selector],
        outputs=[generate_selection_text]
        )
    loudness_selector.change(
        G.change_model_input,
        inputs=[timbre_selector, loudness_selector, f0_selector],
        outputs=[generate_selection_text]
        )
    f0_selector.change(
        G.change_model_input,
        inputs=[timbre_selector, loudness_selector, f0_selector],
        outputs=[generate_selection_text]
        )
  

    source_sample_button.click(
        G.sampel_source_audio_data,
        outputs=[
            source_text,
            source_audio,
            source_image
            ]
    )

    target_sample_button.click(
        G.sampel_target_audio_data,
        outputs=[
            target_text,
            target_audio,
            target_image
            ]
    )

    generate_button.click(
        G.generate_output,
        outputs=[
            rec_audio,
            rec_image
            ]
        )

app.launch(share=True)