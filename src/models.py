import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden, d_in=784, d_out=10):
        super().__init__()

        self.hidden = list(hidden)
        dims = [d_in] + self.hidden + [d_out]

        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x, return_acts=False):
        activations = []

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i < (len(self.layers) - 1):
                x = torch.relu(x)
                activations.append(x)

        return (x, activations) if return_acts else x


def get_params(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def make_net(params, hidden, device):
    model = MLP(hidden).to(device)
    model.load_state_dict(params)
    model.eval()
    return model


def n_hidden_layers(model):
    return len(model.layers) - 1