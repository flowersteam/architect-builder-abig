import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from main_comem.utils.obs_dict import ObsDict


def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class MLPNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, set_final_bias):
        super(MLPNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_outputs)

        if set_final_bias:
            self.fc3.weight.data.mul_(0.1)
            self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


class MLPWordEmbeddingNetwork(nn.Module):
    def __init__(self, obs_space, action_space,
                 hidden_size=128, instr_dim=128, set_final_bias=False):
        super().__init__()
        self.instr_dim = instr_dim
        self.word_embedding = nn.Embedding(obs_space["message"].n, self.instr_dim)
        fc1_input_dim = np.prod(obs_space["tile"].shape) + self.instr_dim
        self.fc1 = nn.Linear(fc1_input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space.n)
        if set_final_bias:
            self.fc3.weight.data.mul_(0.1)
            self.fc3.bias.data.mul_(0.0)

    def forward(self, obs):
        message_embedding = self.word_embedding(obs.get('message'))
        x = torch.cat([message_embedding, obs.get('tile')], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


class MLPAttentionNetwork(nn.Module):
    def __init__(self, obs_space, action_space,
                 hidden_size=128, instr_dim=128, set_final_bias=False):
        super().__init__()
        self.instr_dim = instr_dim
        self.word_embedding = nn.Embedding(obs_space["message"].n, self.instr_dim)
        self.scene_encoder = nn.Linear(np.prod(obs_space["tile"].shape), self.instr_dim)
        self.fc1 = nn.Linear(instr_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, action_space.n)

    def forward(self, obs):
        attention_vec = F.sigmoid(self.word_embedding(obs.get('message')))
        scene_vec = self.scene_encoder(obs.get('tile'))
        x = torch.mul(attention_vec, scene_vec)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

class ConvNetwork(nn.Module):
    """
    Convolutional Network architecture for Pommerman
    Based on: https://arxiv.org/pdf/1812.07297.pdf
    """

    def __init__(self, input_size, num_input_channels, output_vec_size, k=3, p=1):
        super(ConvNetwork, self).__init__()
        self.conv1 = nn.Conv2d(num_input_channels, out_channels=16, kernel_size=k, stride=1, padding=p)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=k, stride=1, padding=p)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=k, stride=1, padding=p)

        self.last_featuremaps_size = (input_size[0] + 3 * (2 * p - k + 1)) * (input_size[1] + 3 * (2 * p - k + 1))
        self.fc = nn.Linear(64 * self.last_featuremaps_size, output_vec_size)

    def forward(self, x):
        if type(x) is ObsDict:
            x = x['obs_map']
        # our obs are batch, x, y, channel but cnn is batch, channel, x, y
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x_flat = x.view(-1, self.fc.in_features)
        out_vector = self.fc(x_flat)
        return out_vector


class FilmLayer(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=imm_channels,
            kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=imm_channels, out_channels=out_features,
            kernel_size=(3, 3), padding=1)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        weight = self.weight(y).unsqueeze(2).unsqueeze(3)
        bias = self.bias(y).unsqueeze(2).unsqueeze(3)
        out = x * weight + bias
        return F.relu(out)


class FiLmCNNNetwork(nn.Module):
    def __init__(self, obs_space, action_space,
                 image_dim=128, instr_dim=128, endpool=True):
        super().__init__()
        self.res = True
        self.image_dim = image_dim
        self.instr_dim = instr_dim
        self.obs_space = obs_space

        self.image_conv = nn.Sequential(*[
            nn.Conv2d(
                in_channels=obs_space['tile'].shape[-1], out_channels=128,
                kernel_size=(3, 3) if endpool else (2, 2), stride=1, padding=1),
            nn.ReLU(),
            *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)])
        ])

        self.film_pool = nn.MaxPool2d(kernel_size=obs_space['tile'].shape[:-1] if endpool else (2, 2), stride=2)

        # Define instruction embedding

        self.word_embedding = nn.Embedding(obs_space["message"].n, self.instr_dim)
        self.final_instr_dim = self.instr_dim

        num_module = 2
        self.controllers = []
        for ni in range(num_module):
            mod = FilmLayer(
                in_features=self.final_instr_dim,
                out_features=128 if ni < num_module - 1 else self.image_dim,
                in_channels=128, imm_channels=128)
            self.controllers.append(mod)
            self.add_module('FiLM_' + str(ni), mod)

        # image embedding
        self.embedding_size = self.image_dim
        self.fc = nn.Linear(self.embedding_size, action_space.n)

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    def forward(self, obs):
        # extract obs and instruction
        message_embedding = self.word_embedding(obs.get('message'))
        x = torch.transpose(torch.transpose(obs.get('tile'), 1, 3), 2, 3)
        x = self.image_conv(x)

        for controller in self.controllers:
            out = controller(x, message_embedding)
            if self.res:
                out += x
            x = out
        x = F.relu(self.film_pool(x))
        x = x.reshape(x.shape[0], -1)

        out = self.fc(x)
        return out
