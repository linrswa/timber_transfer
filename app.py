#%%
import gradio as gr 
from torch.utils.data import DataLoader
import random

from components.timbre_transformer.TimberTransformer import TimbreTransformer
from data.dataset import NSynthDataset

model = TimbreTransformer(is_train=False, is_smooth=True, mlp_layer=3)
dataset = NSynthDataset(data_mode="valid", sr=16000)

def chagne_dataset(data_mode):
    dataset.set_data_mode(data_mode)
    return dataset.data_mode

def generate_data():
    fn_with_path = random.choice(dataset.audio_list)
    return fn_with_path 



with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            data_mode = gr.Radio(["train", "valid", "test"], label="Data Mode")
            data_mode_button = gr.Button("change dataset")
        data_mode_text = gr.Textbox(label="Data Mode")
    
    with gr.Row():
        generate_button = gr.Button("generate data")
        generate_text = gr.Textbox(label="file name")

    data_mode_button.click(chagne_dataset, data_mode, data_mode_text)
    generate_button.click(generate_data, outputs=generate_text)


app.launch()
        
        

    