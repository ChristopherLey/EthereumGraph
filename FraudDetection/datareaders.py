from pathlib import Path

import torch


def get_elliptic_graph(path: Path):
    data_path = path / "elliptic_data.pt"
    assert data_path.exists()
    return torch.load(data_path)
