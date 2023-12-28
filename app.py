#%%
import gradio as gr 
import random
import torch
from numpy import ndarray
from matplotlib import pyplot as plt

from components.timbre_transformer.TimberTransformer import TimbreTransformer
from data.dataset import NSynthDataset

pt_file = "pt_file/New_train_6_generator_best_7.pt"
model = TimbreTransformer(is_train=False, is_smooth=True, mlp_layer=3)
dataset = NSynthDataset(data_mode="train", sr=16000, frequency_with_confidence=True)
model.eval()
model.load_state_dict(torch.load(pt_file))

def np2tensor(np_array: ndarray) -> torch.Tensor:
    return torch.from_numpy(np_array)

def create_fig(data: ndarray) -> plt.Figure:
    fig = plt.figure()
    plt.plot(data)
    plt.close()
    return fig

def cal_loudness_norm(l: ndarray):
    mean_loudness = -41.27331367041325
    std_loudness = 52.82343779478101552
    return (l - mean_loudness) / std_loudness

def model_gen(s: ndarray, l_norm: ndarray, f:ndarray):
    transfrom = lambda x: np2tensor(x).unsqueeze(0)
    s, l_norm, f = transfrom(s), transfrom(l_norm), transfrom(f)
    f = f[:, :-1, 0]
    _, _, rec_s, _, _ = model(s, l_norm, f)
    return rec_s

def sample_data():
    fn_with_path = random.choice(dataset.audio_list)
    fn = fn_with_path.split("/")[-1][:-4]
    _, s, l, f = dataset.getitem_by_fn(fn)
    return fn, s, l, f

def chagne_dataset(data_mode):
    dataset.set_data_mode(data_mode)
    return dataset.data_mode

def generate_data():
    fn, s, l, f = sample_data()
    # create a matplotlib.figure.Figure for s
    fig_s = create_fig(s)
    rec_s = model_gen(s, cal_loudness_norm(l), f).squeeze().detach().numpy()
    fig_rec_s = create_fig(rec_s)
    
    return fn, (16000, s), fig_s, (16000, rec_s), fig_rec_s


with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            data_mode = gr.Radio(["train", "valid", "test"], label="Data Mode")
            data_mode_button = gr.Button("change dataset")
        data_mode_text = gr.Textbox(label="Data Mode")
    
    with gr.Row():
        generate_button = gr.Button("generate data")
        generate_text = gr.Textbox(label="file name")

    with gr.Row():
        generate_image = gr.Plot(label="Ori signal")
        generate_audio = gr.Audio(label="Ori signal")

    with gr.Row():
        generate_rec_image = gr.Plot(label="Rec signal")
        generate_rec_audio = gr.Audio(label="Rec signal")

    data_mode_button.click(chagne_dataset, data_mode, data_mode_text)
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
        

    