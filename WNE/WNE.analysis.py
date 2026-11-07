import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import zscore
from tqdm import tqdm
import warnings
import multiprocessing
from functools import partial





INPUT_DIR = 'step1_data'
NETWORK_FILE = 'converted_network.txt'

OUTPUT_DIR = os.path.join(os.path.dirname(INPUT_DIR), 'step1_result')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"📂 已创建新的输出文件夹: {OUTPUT_DIR}")

STAGES = ['E9.5', 'E10.5', 'E11.5', 'E12.5', 'E13.5', 'E14.5', 'E15.5', 'E16.5']
REF_STAGE = 'E9.5'
ES = 1e-8


def calculate_single_sample_entropy(s_idx, curr_data_bc, ref_data_bc, candidate_genes, adj_map, gene_to_idx):
   
    sample_vec = curr_data_bc[:, s_idx]
    sample_gene_entropy = np.zeros(len(candidate_genes))

    for g_i, gene in enumerate(candidate_genes):
        gene_row_idx = gene_to_idx[gene]
        neighbors = adj_map.get(gene, [])

        if len(neighbors) < 2: continue

        neighbor_indices = [gene_to_idx[n] for n in neighbors]

        center_ref = ref_data_bc[gene_row_idx, :]
        neighbors_ref = ref_data_bc[neighbor_indices, :]

        valid_deltas = []

        for i, n_idx in enumerate(neighbor_indices):
            neigh_ref = ref_data_bc[n_idx, :]


            r_base = np.corrcoef(center_ref, neigh_ref)[0, 1]
            if np.isnan(r_base): r_base = 0
            pcc_base = 0.5 * (1 + abs(r_base))


            c_mix = np.append(center_ref, sample_vec[gene_row_idx])
            n_mix = np.append(neigh_ref, sample_vec[n_idx])

            r_mix = np.corrcoef(c_mix, n_mix)[0, 1]
            if np.isnan(r_mix): r_mix = 0
            pcc_mix_abs = abs(r_mix)


            delta = abs(pcc_mix_abs - pcc_base)
            if delta > 0:
                valid_deltas.append(delta)


        if valid_deltas:
            valid_deltas = np.array(valid_deltas)
            P = valid_deltas / (np.sum(valid_deltas) + ES)
            expr = sample_vec[gene_row_idx]
            term = expr * P
            term_log = np.log(term + ES)
            ent = - (1 / len(valid_deltas)) * np.sum(term * term_log)

            sample_gene_entropy[g_i] = ent

    return sample_gene_entropy



if __name__ == '__main__':
    print(f"🚀 [Server Info] 检测到 CPU 核心数: {os.cpu_count()}")
    NUM_WORKERS = 40
    print(f"⚡ [Acceleration] 将启动 {NUM_WORKERS} 个进程并行计算...")

    print("\n📦 [Step 1] Loading Data...")
    data_dict = {}
    all_samples_list = []

    for stage in STAGES:
        file_path = os.path.join(INPUT_DIR, f'expression_data_{stage}.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing: {file_path}")
        df = pd.read_csv(file_path, index_col=0)
        data_dict[stage] = df
        all_samples_list.append(df)

        print("📊 [Step 2] Global Normalization & Format Fix...")
        full_matrix = pd.concat(all_samples_list, axis=1)
        full_matrix.index = full_matrix.index.str.strip().str.title()  

        norm_values = zscore(full_matrix.values, axis=1)
        full_matrix_norm = pd.DataFrame(
            norm_values,
            index=full_matrix.index,
            columns=full_matrix.columns
        ).abs()

    print("🕸️ [Step 3] Loading Network...")
    try:
        network_df = pd.read_csv(NETWORK_FILE, sep=r'\s+', header=None,
                                 names=['Source', 'Target', 'Score'], dtype={'Source': str, 'Target': str})
    except:
        network_df = pd.read_csv(NETWORK_FILE, sep=r'\s+', header=None,
                                 usecols=[0, 1], names=['Source', 'Target'], dtype=str)

    network_df['Source'] = network_df['Source'].str.strip().str.title()
    network_df['Target'] = network_df['Target'].str.strip().str.title()

    valid_genes = set(full_matrix_norm.index)
    mask = network_df['Source'].isin(valid_genes) & network_df['Target'].isin(valid_genes)
    network_df = network_df[mask]

    adj_map = {}
    for _, row in network_df.iterrows():
        adj_map.setdefault(row['Source'], []).append(row['Target'])

    candidate_genes = [g for g in full_matrix_norm.index if g in adj_map]
    print(f"   -> Valid Center Genes: {len(candidate_genes)}")
    if len(candidate_genes) == 0: exit()

    print("\n🔥 [Step 4] Parallel DNFE Calculation (Running on 40 cores)...")

    ref_samples = data_dict[REF_STAGE].columns
    ref_data_np = full_matrix_norm[ref_samples].values
    gene_to_idx = {gene: i for i, gene in enumerate(full_matrix_norm.index)}

    stage_avg_entropy = []
    dnb_matrix = []

    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        for stage in STAGES:
            print(f"\n👉 Processing Stage: {stage}")

            curr_samples = data_dict[stage].columns
            curr_data_np = full_matrix_norm[curr_samples].values
            n_samples = curr_data_np.shape[1]

            worker_func = partial(
                calculate_single_sample_entropy,
                curr_data_bc=curr_data_np,
                ref_data_bc=ref_data_np,
                candidate_genes=candidate_genes,
                adj_map=adj_map,
                gene_to_idx=gene_to_idx
            )

            results = list(tqdm(pool.imap(worker_func, range(n_samples)),
                                total=n_samples, desc=f"{stage} Samples"))

            stage_results_np = np.array(results)  

            cell_gene_df = pd.DataFrame(stage_results_np, index=curr_samples, columns=candidate_genes)
            cell_gene_csv_path = os.path.join(OUTPUT_DIR, f'Cell_Gene_Entropy_{stage}.csv')
            cell_gene_df.to_csv(cell_gene_csv_path)
            print(f"   💾 已保存该阶段的 [细胞 × 基因] 熵矩阵: {cell_gene_csv_path}")

            avg_gene_entropy = np.mean(stage_results_np, axis=0)

            stage_avg_entropy.append(np.mean(avg_gene_entropy))
            dnb_matrix.append(avg_gene_entropy)

    dnb_matrix = np.array(dnb_matrix).T

    print(f"\n🎨 [Step 5] Saving results to {OUTPUT_DIR}...")

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(STAGES)), stage_avg_entropy, 'r-o', linewidth=3)
    plt.xticks(range(len(STAGES)), STAGES)
    plt.title('Average Entropy Curve')
    plt.xlabel('Stages')
    plt.ylabel('Entropy')
    plt.grid(True, linestyle='--')
    plt.savefig(os.path.join(OUTPUT_DIR, 'Average_Entropy_Curve.png'))
    plt.close()

    top_n = 200
    actual_n = min(top_n, dnb_matrix.shape[0])

    if actual_n > 0:
        max_ent = np.max(dnb_matrix, axis=1)
        top_idx = np.argsort(max_ent)[-actual_n:]

        dnb_subset = dnb_matrix[top_idx, :]

        top_genes_names = np.array(candidate_genes)[top_idx]
        top_200_df = pd.DataFrame(dnb_subset, index=top_genes_names, columns=STAGES)

        top_200_df['Max_Entropy'] = top_200_df.max(axis=1)
        top_200_df = top_200_df.sort_values(by='Max_Entropy', ascending=False)
        top_200_df.drop(columns=['Max_Entropy'], inplace=True)

        top200_csv_path = os.path.join(OUTPUT_DIR, 'Top200_DNB_Genes_Entropy.csv')
        top_200_df.to_csv(top200_csv_path)
        print(f"\n 🧬  [核心结果] 已成功导出 Top {actual_n} DNB 驱动基因至: {top200_csv_path}")

        pd.DataFrame(dnb_matrix, index=candidate_genes, columns=STAGES).to_csv(os.path.join(OUTPUT_DIR, 'All_Candidate_Genes_Entropy.csv'))

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(np.arange(len(STAGES)), np.arange(actual_n))
        surf = ax.plot_surface(X, Y, dnb_subset, cmap='jet')

        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels(STAGES)
        ax.set_title('DNB Landscape')
        fig.colorbar(surf, shrink=0.5)

        plt.savefig(os.path.join(OUTPUT_DIR, 'DNB_Landscape.png'))
        plt.close()

    pd.DataFrame(stage_avg_entropy, index=STAGES, columns=['Entropy']).to_csv(
        os.path.join(OUTPUT_DIR, 'entropy_scores.csv'))

    print("\n✅ 运行完毕！单细胞级基因熵矩阵、绘图等结果均已保存在目标文件夹中。")