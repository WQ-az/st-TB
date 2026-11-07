import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


class SpatiotemporalBifurcationGenerator:

    def __init__(self, n_cells=100, n_genes=50, radius=0.3):
        self.n_cells = n_cells
        self.n_genes = n_genes
        self.radius = radius

    def generate_spatial_graph(self):
        coords = np.random.rand(self.n_cells, 2)
        dist_matrix = squareform(pdist(coords, metric='euclidean'))
        adjacency = (dist_matrix <= self.radius).astype(float)
        np.fill_diagonal(adjacency, 0)

        degree = adjacency.sum(axis=1, keepdims=True)
        return coords, adjacency, degree

    def _get_higher_order_terms(self, u, scale=0.01):
        noise = np.zeros_like(u)
        for d in range(3, 7):
            alpha = np.random.randn() * scale
            noise += alpha * (u ** d)
        return noise

    def reaction_term(self, state, mu, bifurcation_type):
        u = state[:, 0]

        if bifurcation_type == 'null':
            ho = self._get_higher_order_terms(u, scale=0.005)
            return np.column_stack([-u + ho])

        elif bifurcation_type == 'pitchfork':
            ho = self._get_higher_order_terms(u, scale=0.01)
            return np.column_stack([mu * u - u**3 + ho])

        elif bifurcation_type == 'fold':
            ho = self._get_higher_order_terms(u, scale=0.01)
            return np.column_stack([mu + u**2 + ho])

        elif bifurcation_type == 'transcritical':
            ho = self._get_higher_order_terms(u, scale=0.01)
            return np.column_stack([mu * u - u**2  + ho])

        elif bifurcation_type == 'hopf':
            v = state[:, 1]
            r_squared = u**2 + v**2
            ho_u = self._get_higher_order_terms(u, scale=0.01)
            ho_v = self._get_higher_order_terms(v, scale=0.01)
            f_u = mu * u - v - u * r_squared + ho_u
            f_v = mu * v + u - v * r_squared + ho_v
            return np.column_stack([f_u, f_v])

        elif bifurcation_type == 'period_doubling':
            v = state[:, 1]
            w = state[:, 2]
            a, b = 0.2, 0.2
            c = 5.0 + mu
            f_u = -v - w
            f_v = u + a * v
            f_w = b + w * (u - c)
            return np.column_stack([f_u, f_v, f_w])

    def simulate_spatial_sde(self, bifurcation_type, T_max=1000, dt=0.05, D=0.1, sigma=0.05):
        coords, adjacency, degree = self.generate_spatial_graph()

        dims = 2 if bifurcation_type == 'hopf' else (3 if bifurcation_type == 'period_doubling' else 1)
        state = np.random.randn(self.n_cells, dims) * 0.01

        if bifurcation_type == 'fold':
            mu_0 = np.random.uniform(-0.25, -0.05)
            state[:, 0] = -np.sqrt(np.abs(mu_0)) + np.random.randn(self.n_cells) * 0.01
        elif bifurcation_type == 'period_doubling':
            mu_0 = np.random.uniform(-2.5, -1.5)
        else:
            mu_0 = np.random.uniform(-1.0, -0.2)

        mu_values = np.full(T_max, mu_0) if bifurcation_type == 'null' else np.linspace(mu_0, 0.0, T_max)

        for _ in range(100):
            f = self.reaction_term(state, mu_0, bifurcation_type)
            spatial_coupling = D * (adjacency @ state - degree * state)
            noise = sigma * np.random.randn(self.n_cells, dims) * np.sqrt(dt)
            state = state + (f + spatial_coupling) * dt + noise

        trajectory_u = []
        recorded_mu = []

        for t in range(T_max):
            mu = mu_values[t]
            f = self.reaction_term(state, mu, bifurcation_type)

            spatial_coupling = D * (adjacency @ state - degree * state)

            noise = sigma * np.random.randn(self.n_cells, dims) * np.sqrt(dt)
            state = state + (f + spatial_coupling) * dt + noise

            trajectory_u.append(state[:, 0].copy())
            recorded_mu.append(mu)

            max_deviation = 0
            u = state[:, 0]

            if bifurcation_type in ['null', 'pitchfork', 'transcritical']:
                max_deviation = np.abs(u).max()
            elif bifurcation_type == 'fold':
                eq = -np.sqrt(np.abs(mu)) if mu < 0 else 0
                max_deviation = np.abs(u - eq).max()
            elif bifurcation_type == 'hopf':
                v = state[:, 1]
                max_deviation = np.sqrt(u**2 + v**2).max()
            elif bifurcation_type == 'period_doubling':
                if np.abs(u).max() > 50.0:
                    break

            if bifurcation_type != 'period_doubling' and max_deviation > 10 * sigma:
                break

        trajectory_u = np.array(trajectory_u)
        recorded_mu = np.array(recorded_mu)
        actual_T = len(trajectory_u)

        n_samples = min(50, actual_T)
        if n_samples < 15:
            return self.simulate_spatial_sde(bifurcation_type, T_max, dt, D, sigma)

        early_samples = np.linspace(0, actual_T//2, n_samples//3, dtype=int, endpoint=False)
        late_samples = np.linspace(actual_T//2, actual_T-1, 2*n_samples//3, dtype=int)
        sampled_indices = np.unique(np.concatenate([early_samples, late_samples]))

        return trajectory_u[sampled_indices], sampled_indices * dt, recorded_mu[sampled_indices], coords, adjacency

    def map_to_high_dim(self, trajectory_u):
        T, N, dims = trajectory_u.shape

        W1 = np.random.randn(dims, 16) / np.sqrt(dims)
        W2 = np.random.randn(16, self.n_genes) / np.sqrt(16)

        hidden = np.tanh(trajectory_u @ W1)
        gene_expr = hidden @ W2

        dropout_mask = (np.random.rand(T, N, self.n_genes) > 0.15).astype(float)
        gaussian_noise = np.random.randn(T, N, self.n_genes) * 0.05

        expression = gene_expr * dropout_mask + gaussian_noise

        return expression

    def generate_sample(self, bifurcation_type):
        trajectory_u, time_stamps, mu_traj, coords, adjacency = self.simulate_spatial_sde(bifurcation_type)

        if len(trajectory_u.shape) == 2:
            trajectory_u = trajectory_u[:, :, np.newaxis]

        expression = self.map_to_high_dim(trajectory_u)
        edge_index = np.array(np.where(adjacency > 0))
        label_map = {'null': 0, 'pitchfork': 1, 'fold': 2, 'transcritical': 3, 'period_doubling': 4, 'hopf': 5}

        return {
            'x': expression,
            'edge_index': edge_index,
            't': time_stamps,
            'y': label_map[bifurcation_type],
            'coords': coords,
            'mu_trajectory': mu_traj
        }

    def generate_dataset(self, n_samples_per_class=2000, save_path=None):
        bifurcation_types = ['null', 'pitchfork', 'fold', 'transcritical', 'period_doubling', 'hopf']
        dataset = []

        print("开始生成数据集...")
        for bif_type in bifurcation_types:
            print(f"\n生成 {bif_type} 分岔样本: {n_samples_per_class} 个")
            for i in range(n_samples_per_class):
                if (i+1) % 500 == 0:
                    print(f"  进度: {i+1}/{n_samples_per_class}")

                sample = self.generate_sample(bif_type)
                dataset.append(sample)

        print(f"\n数据集生成完成！总样本数: {len(dataset)}")

        if save_path:
            np.savez_compressed(save_path, dataset=dataset)
            print(f"数据已保存到: {save_path}")

        return dataset


if __name__ == "__main__":
    generator = SpatiotemporalBifurcationGenerator(n_cells=100, n_genes=50, radius=0.3)
    sample = generator.generate_sample('pitchfork')

    print("=" * 60)
    print("测试单个样本生成成功 ")
    print(f"x shape: {sample['x'].shape}, y: {sample['y']}")
    print("=" * 60)