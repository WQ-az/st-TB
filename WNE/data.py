import scanpy as sc
import pandas as pd
import numpy as np
import os

file_path = 'Mouse_embryo_all_stage.h5ad'
output_dir = 'step1_data'  
stage_col = 'timepoint'

TARGET_GENES = 2000  
TARGET_CELLS = 50000

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f"正在加载数据: {file_path} ...")
adata = sc.read_h5ad(file_path)
print(f"原始数据形状: {adata.shape}")

print("\n🧬 [第一步] 动态筛选基因...")
genes_n_cells = (adata.X > 0).sum(axis=0)
if hasattr(genes_n_cells, "A1"):  
    genes_n_cells = genes_n_cells.A1
else:
    genes_n_cells = np.array(genes_n_cells).flatten()
sorted_counts = np.sort(genes_n_cells)[::-1]

if len(sorted_counts) > TARGET_GENES:
    gene_cutoff = sorted_counts[TARGET_GENES - 1]
    print(f"   -> 目标保留 {TARGET_GENES} 个基因。")
    print(f"   -> 自动计算出的 min_cells 阈值为: {gene_cutoff}")
    sc.pp.filter_genes(adata, min_cells=gene_cutoff)
else:
    print(f"   -> 现有基因数 ({len(sorted_counts)}) 少于目标，不进行基因过滤。")
print(f"   -> 基因过滤后形状: {adata.shape}")


print("\n🔄 [第二步] 基因变动，重新计算细胞 QC 指标...")

sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)


print("\n🔬 [第三步] 动态筛选样本 (优中选优)...")

current_total_cells = adata.n_obs
retention_ratio = TARGET_CELLS / current_total_cells
print(f"   -> 当前样本: {current_total_cells}, 目标: {TARGET_CELLS}")
print(f"   -> 全局保留比例: {retention_ratio:.2%}")

final_cells_to_keep = []
stages = adata.obs[stage_col].unique()

print(f"\n   -> 开始按阶段筛选 (保留各阶段前 {retention_ratio:.1%} 的高质量细胞):")
print(f"   {'阶段':<10} | {'原始数量':<10} | {'保留数量':<10} | {'最低基因数门槛'}")
print("-" * 60)

for stage in stages:
  
    stage_obs = adata.obs[adata.obs[stage_col] == stage]
    n_stage_total = len(stage_obs)  
    n_keep = int(n_stage_total * retention_ratio)  
    if n_keep < 1: n_keep = 1   
    top_cells = stage_obs.sort_values(by='n_genes_by_counts', ascending=False).head(n_keep)
    final_cells_to_keep.extend(top_cells.index.tolist())    
    min_gene_threshold = top_cells['n_genes_by_counts'].min()

    print(f"   {stage:<10} | {n_stage_total:<10} | {n_keep:<10} | > {min_gene_threshold:.0f}")

adata = adata[final_cells_to_keep, :].copy()

print("-" * 60)
print(f"✅ 最终数据形状: {adata.shape}")

print("\n💾 [第四步] 开始导出文件...")
for stage in adata.obs[stage_col].unique():
    subset = adata[adata.obs[stage_col] == stage].copy()
    metadata = pd.DataFrame(index=subset.obs_names)
    metadata['stage'] = subset.obs[stage_col]
    if 'spatial' in subset.obsm.keys():
        metadata['spatial_x'] = subset.obsm['spatial'][:, 0]
        metadata['spatial_y'] = subset.obsm['spatial'][:, 1]

    metadata.to_csv(os.path.join(output_dir, f'metadata_spatial_{stage}.csv'), index_label='sample_id')

    if 'count' in subset.layers:
        X_data = subset.layers['count']
    else:
        X_data = subset.X
    if hasattr(X_data, "toarray"):
        X_data = X_data.toarray()

    exp_df = pd.DataFrame(X_data.T, index=subset.var_names, columns=subset.obs_names)
    exp_df.to_csv(os.path.join(output_dir, f'expression_data_{stage}.csv'), index_label='gene_name')

    print(f"   -> 已导出: {stage}")

print("\n🎉 全部完成！")