import torch
from torch.utils.data import DataLoader, TensorDataset


def get_data(name, eval_n, device, root="./data"):

    from torchvision import datasets, transforms

    transform = transforms.ToTensor()
    dataset = getattr(datasets, name)

    train_data = dataset(root=root, train=True, download=True, transform=transform)
    test_data = dataset(root=root, train=False, download=True, transform=transform)

    def to_xy(data):
        X = data.data.float().view(len(data), -1)/255.0
        y = data.targets.long()
        return X, y

    Xtr, ytr = to_xy(train_data)
    Xte, yte = to_xy(test_data)

    indices = torch.randperm(len(Xte))[:eval_n]

    Xev = Xte[indices].to(device)
    yev = yte[indices].to(device)

    return Xtr, ytr, Xev, yev


def loader(Xtr, ytr, batch_size):
    dataset = TensorDataset(Xtr, ytr)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)