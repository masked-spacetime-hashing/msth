import numpy as np
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

df = px.data.tips()
fig = px.histogram(df, x="total_bill")


def gen_spatial_grid(reso=256):
    """3d uniform spatial grid in [0,1]^3"""
    x = torch.linspace(0, 1, reso)
    y = torch.linspace(0, 1, reso)
    z = torch.linspace(0, 1, reso)
    x, y, z = torch.meshgrid(x, y, z)
    return torch.stack([x, y, z], dim=-1).reshape(-1, 3)


def viz_histograms(xs, names=None, show=True):
    _num = len(xs)
    xs = [x.flatten().cpu().numpy() for x in xs]
    fig = make_subplots(rows=_num, cols=1, shared_yaxes=True, subplot_titles=names)
    hists = []
    for x in xs:
        counts, bins = np.histogram(x, bins=100)
        bins = 0.5 * (bins[:-1] + bins[1:])
        hists.append(go.Bar(x=bins, y=counts))
    fig.add_traces(hists, rows=list(range(1, _num + 1)), cols=[1] * _num)

    if show:
        fig.show()


@torch.no_grad()
def hist_from_mask(mask, reso=256, chunk_size=1 << 17):
    grid = gen_spatial_grid(reso)
    tot_size = grid.size(0)
    vals = torch.zeros([tot_size, 1]).to()

    for start in range(0, tot_size, chunk_size):
        end = min(start + chunk_size, tot_size)
        vals[start:end] = mask(grid[start:end])[..., 0:1].to()


def viz_distribution(dists, show=True):
    dists = dists.cpu().numpy()
    _num = dists.shape[0]
    dists = [dist.flatten() for dist in dists]
    fig = make_subplots(rows=_num, cols=1, shared_yaxes=True)

    hists = []
    for x in dists:
        counts, bins = np.histogram(x, bins=100)
        bins = 0.5 * (bins[:-1] + bins[1:])
        hists.append(go.Bar(x=bins, y=counts))
    fig.add_traces(hists, rows=list(range(1, _num + 1)), cols=[1] * _num)

    if show:
        fig.show()
