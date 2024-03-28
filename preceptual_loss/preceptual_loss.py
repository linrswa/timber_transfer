#%%
import torch
import torch.nn as nn
import os 


class ConvBlock(nn.Module):
    def __init__(self, f, w, s, in_channels):
        super().__init__()
        p1 = (w - 1) // 2
        p2 = (w - 1) - p1
        self.pad = nn.ZeroPad2d((0, 0, p1, p2))

        self.conv2d = nn.Conv2d(
            in_channels=in_channels, out_channels=f, kernel_size=(w, 1), stride=s
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(f)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CREPE(nn.Module):
    def __init__(self, model_capacity="full"):
        super().__init__()

        capacity_multiplier = {"tiny": 4, "small": 8, "medium": 16, "large": 24, "full": 32}[
            model_capacity
        ]

        self.layers = [1, 2, 3, 4, 5, 6]
        filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        filters = [1] + filters
        widths = [512, 64, 64, 64, 64, 64]
        strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

        for i in range(len(self.layers)):
            f, w, s, in_channel = filters[i + 1], widths[i], strides[i], filters[i]
            self.add_module("conv%d" % i, ConvBlock(f, w, s, in_channel))

        self.linear = nn.Linear(64 * capacity_multiplier, 360)
        self.load_weight(model_capacity)
        self.eval()

    def load_weight(self, model_capacity):
        package_dir = os.path.dirname(os.path.realpath(__file__))
        filename = "crepe-{}.pth".format(model_capacity)
        self.load_state_dict(torch.load(os.path.join(package_dir, filename)))

    def forward(self, x):
        # x : shape (batch, sample)
        x = x.view(x.shape[0], 1, -1, 1)
        for i in range(len(self.layers)):
            x = self.__getattr__("conv%d" % i)(x)
            if i == 4: preceptual_feature = x
            
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return preceptual_feature


# write a function to divide the signal to frames.
def frame_signal(signal, frame_length=1024):

    # Pad signal to ensure it can be evenly divided into frames
    signal = torch.nn.functional.pad(signal, (0, frame_length - signal.shape[-1] % frame_length), 'constant', 0)

    # Reshape signal to split it into frames
    signal = signal.reshape(signal.shape[0], -1, frame_length)

    # Calculate the number of frames
    num_frames = signal.shape[1]

    return signal, num_frames

# # FIXME: try to make this function accept batch input
# def get_preceptual_feature(signals, frame_length=1024):
#     cr = CREPE('small').to(signals.device)
#     signals, _ = frame_signal(signals, frame_length=frame_length)
#     c = 0
#     for s in signals:
#         # s.shape = [63(frames), 1024]
#         if c > 0:
#             pf = torch.cat((pf, cr(s)), dim=0)
#         else:
#             pf = cr(s) # [63(frames), 64, 8, 1]
#             c += 1
        
#     return pf # [2016(32*63), 64, 8, 1]

def get_preceptual_feature(signals, frame_length=1024):
    cr = CREPE('small').to(signals.device)
    framed_signals, _ = frame_signal(signals, frame_length=frame_length)
    batch = framed_signals.shape[0]
    framed_signals = framed_signals.view(-1, frame_length)
    pf = cr(framed_signals)
    return pf

def cal_preceptual_loss(signal, target):
    # pf.sahpe = [batch, 1, 64000], target.shape = [batch, 64000, 1]
    pf = get_preceptual_feature(signal)
    pf_target = get_preceptual_feature(target.permute(0, 2, 1))
    loss = nn.L1Loss()(pf, pf_target)
    return loss
