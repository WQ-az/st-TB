
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from pathlib import Path
from typing import Tuple, List


class BifurcationDataset(Dataset):


    LABEL_MAP = {
        'null': 0,
        'pitchfork': 1,
        'fold': 2,
        'transcritical': 3,
        'period_doubling': 4,
        'hopf': 5
    }
    

    CLASS_NAMES = ['Null', 'Pitchfork', 'Fold', 'Transcritical', 'Period_doubling', 'Hopf']

    def __init__(self, data_path: str, normalize: bool = True):

        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        raw = np.load(data_path, allow_pickle=True)
        self.samples: List[dict] = list(raw['dataset'])
        self.normalize = normalize


        if self.normalize:
            self._precompute_stats()

    def _precompute_stats(self):

        all_x = []
        for s in self.samples:
            all_x.append(s['x'])  

        all_x = np.concatenate([x.reshape(-1, x.shape[-1]) for x in all_x], axis=0)
        self.global_mean = torch.tensor(all_x.mean(axis=0), dtype=torch.float32)
        self.global_std = torch.tensor(all_x.std(axis=0) + 1e-6, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        s = self.samples[idx]

        x = torch.tensor(s['x'], dtype=torch.float32)                   
        edge_index = torch.tensor(s['edge_index'], dtype=torch.long)   
        t = torch.tensor(s['t'], dtype=torch.float32)                  
        y = torch.tensor(s['y'], dtype=torch.long)                     
        coords = torch.tensor(s['coords'], dtype=torch.float32)         

        if self.normalize:
      
            x = (x - self.global_mean) / self.global_std

        mu = torch.tensor(s['mu_trajectory'], dtype=torch.float32)     
        T, N, G = x.shape


        if y.item() == 0:
            tcp_label = torch.zeros(T, dtype=torch.float32)
        else:
            sigma = 0.15
            tcp_label = torch.exp(-torch.abs(mu) / sigma)


        if y.item() == 0:
            scm_label = torch.zeros(N, dtype=torch.float32)
        else:
            x_last = x[-1]  
            mean_expr = x_last.mean(dim=0, keepdim=True)  
            deviation = torch.norm(x_last - mean_expr, dim=1)  

            p95 = torch.quantile(deviation, 0.95)
            scm_label = torch.clamp(deviation / (p95 + 1e-6), 0.0, 1.0)

        return Data(
            x=x, edge_index=edge_index, t=t, y=y, coords=coords,
            tcp_label=tcp_label, scm_label=scm_label
        )

    @staticmethod
    def _zscore(x: torch.Tensor) -> torch.Tensor:

        mean = x.mean(dim=(0, 1), keepdim=True)
        std  = x.std(dim=(0, 1), keepdim=True) + 1e-6
        return (x - mean) / std


def custom_collate(data_list: List[Data]) -> Batch:
 
    if len(data_list) == 0:
        raise ValueError("Empty data_list")


    max_T = max([d.x.shape[0] for d in data_list])
    G = data_list[0].x.shape[2]  

    x_list = []
    edge_index_list = []
    y_list = []
    coords_list = []
    batch_list = []  

    node_offset = 0  

    for graph_idx, data in enumerate(data_list):
        T_current = data.x.shape[0]
        N = data.x.shape[1] 


        x_transposed = data.x.permute(1, 0, 2)  
        if T_current < max_T:
            pad_length = max_T - T_current
            last_frame = x_transposed[:, -1:, :]  
            pad = last_frame.repeat(1, pad_length, 1) 
            x_padded = torch.cat([x_transposed, pad], dim=1)
        else:
            x_padded = x_transposed
        x_list.append(x_padded)  


        edge_index_offset = data.edge_index + node_offset
        edge_index_list.append(edge_index_offset)

        y_list.append(data.y)

        coords_list.append(data.coords)

        batch_list.append(torch.full((N,), graph_idx, dtype=torch.long))

        node_offset += N

    batch_x = torch.cat(x_list, dim=0)
    batch_edge_index = torch.cat(edge_index_list, dim=1)
    batch_y = torch.stack(y_list)
    batch_coords = torch.cat(coords_list, dim=0)
    batch_idx = torch.cat(batch_list, dim=0)


    t_first = data_list[0].t
    if len(t_first) < max_T:
        t_padded = torch.cat([
            t_first,
            torch.full((max_T - len(t_first),), t_first[-1].item(), dtype=t_first.dtype)
        ])
    else:
        t_padded = t_first[:max_T]


    node_counts = [d.x.shape[1] for d in data_list]  
    ptr = torch.tensor([0] + node_counts).cumsum(0)  


    tcp_list = []
    scm_list = []
    for data in data_list:
        tcp = data.tcp_label  
        scm = data.scm_label  
        if len(tcp) < max_T:
            tcp_pad = torch.cat([tcp, torch.zeros(max_T - len(tcp), dtype=tcp.dtype)])
        else:
            tcp_pad = tcp[:max_T]
        tcp_list.append(tcp_pad)
        scm_list.append(scm)
    batch_tcp = torch.stack(tcp_list) 
    batch_scm = torch.cat(scm_list)    

    batch = Batch(
        x=batch_x,
        edge_index=batch_edge_index,
        y=batch_y,
        coords=batch_coords,
        batch=batch_idx,
        t=t_padded,
        ptr=ptr,
        tcp_label=batch_tcp,
        scm_label=batch_scm
    )


    batch._num_graphs = len(data_list)

    return batch


def build_dataloaders(
    data_dir: str,
    batch_size: int = 32, 
    num_workers: int = 4,
    pin_memory: bool = True,
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    data_dir = Path(data_dir)

    train_ds = BifurcationDataset(data_dir / 'train.npz', normalize=normalize)
    val_ds   = BifurcationDataset(data_dir / 'val.npz',   normalize=normalize)
    test_ds  = BifurcationDataset(data_dir / 'test.npz',  normalize=normalize)


    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=pin_memory, collate_fn=custom_collate)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=pin_memory, collate_fn=custom_collate)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=pin_memory, collate_fn=custom_collate)

    return train_loader, val_loader, test_loader