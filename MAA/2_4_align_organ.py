


import os
import argparse
import warnings
from scipy.ndimage import gaussian_filter
warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np

from skimage.morphology import disk
from scipy.ndimage import distance_transform_edt, binary_closing, binary_dilation, gaussian_filter

from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt
from skimage.metrics import structural_similarity as ssim


N_THREADS = 40
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[k] = str(N_THREADS)

from Preprocessing import PreprocessConfig, STIntegrator
from Alignment import align_stack_chain, LDDMM
from RGBgenerator import (
    create_rgb_image_with_mapping,
    interpolate_zero_pixels,
    direct_image_mapping_with_coords,
)

DATA_PATH = "Mouse_embryo_all_stage.h5ad"
BASE_OUT = "output"
ALIGNED_OUT = os.path.join(BASE_OUT, "aligned")


TIMEPOINTS = ["E9.5", "E10.5", "E11.5", "E12.5", "E13.5", "E14.5", "E15.5", "E16.5"]



LDDMM_PARAM_PROFILES = {

    "Brain": {
        "early": {"T": 32, "K": 200, "sigma": 0.8, "alpha": 3.0, "gamma": 3.0, "epsilon": 0.01},
        "late":  {"T": 32, "K": 150, "sigma": 1.0, "alpha": 3.0, "gamma": 3.0, "epsilon": 0.01},
    },

    "Lung":  {
        "early": {"T": 32, "K": 200, "sigma": 0.8, "alpha": 1.5, "gamma": 1.5, "epsilon": 0.01},
        "late":  {"T": 32, "K": 150, "sigma": 1.0, "alpha": 2.0, "gamma": 1.5, "epsilon": 0.01},
    },

}


def ensure_dirs():
    os.makedirs(ALIGNED_OUT, exist_ok=True)



def extract_organ(organ: str):

    print(f"[{organ}] 读取全胚胎数据: {DATA_PATH}")
    adata_all = ad.read_h5ad(DATA_PATH, backed="r")
    adatas = []
    actual = []
    for t in TIMEPOINTS:
        mask = (adata_all.obs["timepoint"] == t) & (adata_all.obs["annotation"] == organ)
        n = int(mask.sum())
        if n > 0:
            ad_sub = adata_all[mask].to_memory()
            adatas.append(ad_sub)
            actual.append(t)
            print(f"  {t}: {n} spots")
    adata_all.file.close()
    if len(adatas) < 2:
        raise ValueError(f"{organ} 可用于对齐的时期不足 2 个，实际仅 {len(adatas)} 个")
    return adatas, actual


def preprocess(adatas, actual):

    cfg = PreprocessConfig(
        integrate="combat",
        use_hvg=True,
        n_top_genes=2000,
        n_pcs=30,
        num_threads=N_THREADS,
        seed=42,
    )
    integrator = STIntegrator(cfg)
    merged = integrator.preprocess_and_integrate(adatas, batch_names=actual)
    emb, rgb01, split, split_rgb = integrator.embed(merged, method="umap", n_components=3)
    for i, a in enumerate(adatas):
        a.obsm["feat"] = split[actual[i]].astype(np.float32)
        a.obsm["feat_rgb"] = split_rgb[actual[i]].astype(np.float32)
    return adatas


def global_chain_align(adatas, actual, organ: str):

    print(f"[{organ}] 开始全局链式对齐 (mode=G2, reference=最晚期)...")

    trim_dict = { "Brain": 0.10, "Lung": 0.07}
    trim_frac = trim_dict.get(organ, 0.10)

    order = list(range(len(adatas) - 1, -1, -1))
    adatas_aligned, params, start_id = align_stack_chain(
        adatas,
        order=order,
        mode="G2",
        feat_key="feat",
        xy_keys=["spatial"] * len(adatas),
        trim_fraction=trim_frac,
        lam=1e-2,
        if_labeled=False,
        verbose=True,
    )
    for a in adatas_aligned:
        a.obsm["spatial_aligned"] = a.obsm["spatial_aligned"].astype(np.float32)
    print(f"[{organ}] 全局对齐完成")
    return adatas_aligned, params, start_id


def _rasterize_mask_sdt(coords, img_size, bbox=None, n_obs=None):

    coords = np.asarray(coords, dtype=float)
    H, W = int(img_size[0]), int(img_size[1])

    if bbox is None:
        xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
        ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
    else:
        xmin, xmax, ymin, ymax = bbox

    x_scaled = np.interp(coords[:, 0], (xmin, xmax), (0, W - 1)).astype(int)
    y_scaled = np.interp(coords[:, 1], (ymin, ymax), (0, H - 1)).astype(int)
    x_scaled = np.clip(x_scaled, 0, W - 1)
    y_scaled = np.clip(y_scaled, 0, H - 1)

    mask = np.zeros((H, W), dtype=np.float32)
    mask[y_scaled, x_scaled] = 1.0

    if n_obs is not None and n_obs > 0:
 
        expected_dist = np.sqrt((H * W) / n_obs)

        radius = int(np.ceil(expected_dist / 1.5))
    else:
        radius = 5
    radius = min(max(radius, 2), 15)  
    solid_mask = binary_closing(mask, structure=disk(radius))
    solid_mask = binary_dilation(solid_mask, structure=disk(max(1, radius // 2))).astype(np.float32)
    dt_in = distance_transform_edt(solid_mask)
    dt_out = distance_transform_edt(1.0 - solid_mask)
    sdt = dt_in - dt_out
    sdt = sdt / (np.abs(sdt).max() + 1e-8)
    return solid_mask[..., np.newaxis], sdt[..., np.newaxis]


def lddmm_refine(adatas_aligned, actual, start_id, organ: str, img_size: int, lddmm_params: dict, sdt_weight: float = 1.0):

    print(f"[{organ}] 开始 LDDMM 局部非线性精配准 (img_size={img_size})...")
    lddmm_model = LDDMM()
    adatas_aligned[start_id].obsm["spatial_lddmm"] = adatas_aligned[start_id].obsm["spatial_aligned"].copy()

    align_order = list(range(len(adatas_aligned) - 1, -1, -1))
    jac_records = []


    is_staged = "early" in lddmm_params and "late" in lddmm_params

    for i in range(1, len(align_order)):
        k_tgt = align_order[i - 1]
        k_src = align_order[i]
        tgt_adata = adatas_aligned[k_tgt]
        src_adata = adatas_aligned[k_src]
        t_src = actual[k_src]
        t_tgt = actual[k_tgt]
        print(f"  LDDMM: {t_src} -> {t_tgt}  (src={src_adata.n_obs}, tgt={tgt_adata.n_obs})")

        t_src_num = float(t_src.replace("E", ""))
        if is_staged:
            threshold = 15.5 if organ == "Lung" else 13.5
            if t_src_num >= threshold:

                curr_lddmm = lddmm_params["late"]
                if organ == "Lung":
                    curr_sdt_weight = sdt_weight * 0.5  
                elif organ == "Brain":
                    curr_sdt_weight = sdt_weight * 0.3
                else:
                    curr_sdt_weight = sdt_weight * 1.2
            else:
                curr_lddmm = lddmm_params["early"]
                if organ == "Brain":
                    curr_sdt_weight = sdt_weight * 0.3

                elif organ == "Lung":
                    curr_sdt_weight = sdt_weight * 0.2  
                else:
                    curr_sdt_weight = sdt_weight
        else:
            curr_lddmm = lddmm_params
            curr_sdt_weight = sdt_weight
        print(f"    使用参数: {curr_lddmm}, SDT权重: {curr_sdt_weight:.3f}")

        curr_img_size = img_size
        max_obs = max(src_adata.n_obs, tgt_adata.n_obs)


        print(f"    [自适应] 当前局部配准分辨率调整为: {curr_img_size}x{curr_img_size}")
  
        src_xy = np.asarray(src_adata.obsm["spatial_aligned"], dtype=float)
        tgt_xy = np.asarray(tgt_adata.obsm["spatial_lddmm"], dtype=float)

        xmin = min(src_xy[:, 0].min(), tgt_xy[:, 0].min())
        xmax = max(src_xy[:, 0].max(), tgt_xy[:, 0].max())
        ymin = min(src_xy[:, 1].min(), tgt_xy[:, 1].min())
        ymax = max(src_xy[:, 1].max(), tgt_xy[:, 1].max())


        x_range = xmax - xmin
        y_range = ymax - ymin
        max_range = max(x_range, y_range)

        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0

        xmin = x_center - max_range / 2.0
        xmax = x_center + max_range / 2.0
        ymin = y_center - max_range / 2.0
        ymax = y_center + max_range / 2.0


        pair_bbox = (xmin, xmax, ymin, ymax)

        img_tgt, mapping_tgt = create_rgb_image_with_mapping(
            tgt_xy, tgt_adata.obsm["feat"], img_size=(curr_img_size, curr_img_size), bbox=pair_bbox
        )
        img_src, mapping_src = create_rgb_image_with_mapping(
            src_xy, src_adata.obsm["feat"], img_size=(curr_img_size, curr_img_size), bbox=pair_bbox
        )

        img_tgt_smooth = interpolate_zero_pixels(img_tgt, radius=2)
        img_src_smooth = interpolate_zero_pixels(img_src, radius=2)


        if organ in ("Lung", "Brain"):
            mask_tgt, sdt_tgt = _rasterize_mask_sdt(
                tgt_xy, (curr_img_size, curr_img_size), bbox=pair_bbox, n_obs=tgt_adata.n_obs
            )
            mask_src, sdt_src = _rasterize_mask_sdt(
                src_xy, (curr_img_size, curr_img_size), bbox=pair_bbox, n_obs=src_adata.n_obs
            )
            img_tgt_smooth = np.concatenate([img_tgt_smooth, mask_tgt, sdt_tgt], axis=2)
            img_src_smooth = np.concatenate([img_src_smooth, mask_src, sdt_src], axis=2)
            print(f"    Added solid mask + SDT channels: shape {img_src_smooth.shape}")


        n_channels = img_src_smooth.shape[2]

        area = curr_img_size * curr_img_size
        dist_src = np.sqrt(area / max(src_adata.n_obs, 1))
        dist_tgt = np.sqrt(area / max(tgt_adata.n_obs, 1))

        dynamic_sigma = min(max(dist_src * 0.6, dist_tgt * 0.6, 3.0), 15.0)
        print(f"    [Sparsity Adapt] Dynamic Gaussian Sigma: {dynamic_sigma:.2f}")

        for c in range(min(3, n_channels)):
            img_src_smooth[..., c] = gaussian_filter(img_src_smooth[..., c], sigma=dynamic_sigma)
            img_tgt_smooth[..., c] = gaussian_filter(img_tgt_smooth[..., c], sigma=dynamic_sigma)

        for c in range(n_channels):

            src_min, src_max = img_src_smooth[..., c].min(), img_src_smooth[..., c].max()
            tgt_min, tgt_max = img_tgt_smooth[..., c].min(), img_tgt_smooth[..., c].max()
            src_range = src_max - src_min + 1e-8
            tgt_range = tgt_max - tgt_min + 1e-8
            img_src_smooth[..., c] = (img_src_smooth[..., c] - src_min) / src_range
            img_tgt_smooth[..., c] = (img_tgt_smooth[..., c] - tgt_min) / tgt_range


        if organ in ("Lung", "Brain"):
            mask_channel_idx = 3
            sdt_channel_idx = 4
            img_src_smooth[..., mask_channel_idx] *= curr_sdt_weight
            img_tgt_smooth[..., mask_channel_idx] *= curr_sdt_weight
            img_src_smooth[..., sdt_channel_idx] *= curr_sdt_weight
            img_tgt_smooth[..., sdt_channel_idx] *= curr_sdt_weight

        J0_final, v_hat, energies, length, Phi0, Phi1, J0, J1 = lddmm_model.register(
            img_src_smooth, img_tgt_smooth, **curr_lddmm
        )

        detPhi1 = lddmm_model.jacobian_derterminant(Phi1)
        jac_min = float(detPhi1.min())
        jac_max = float(detPhi1.max())
        jac_mean = float(detPhi1.mean())
        jac_pct_neg = float((detPhi1 < 0).mean() * 100.0)
        jac_folding = bool(lddmm_model.is_injectivity_violated(detPhi1))
        jac_records.append(
            {
                "pair": f"{t_src}->{t_tgt}",
                "src_stage": t_src,
                "tgt_stage": t_tgt,
                "jac_min": jac_min,
                "jac_max": jac_max,
                "jac_mean": jac_mean,
                "jac_pct_neg": jac_pct_neg,
                "jac_folding": jac_folding,
            }
        )
        print(f"    Jacobian: min={jac_min:.4f}, mean={jac_mean:.4f}, neg%={jac_pct_neg:.4f}, folding={jac_folding}")

        mapped_img, new_cell_coords_grid = direct_image_mapping_with_coords(img_src, Phi0[-1], mapping_src)
        new_cell_coords = new_cell_coords_grid.astype(np.float64)
        new_cell_coords[:, 0] = np.interp(new_cell_coords[:, 0], (0, curr_img_size - 1), (xmin, xmax))
        new_cell_coords[:, 1] = np.interp(new_cell_coords[:, 1], (0, curr_img_size - 1), (ymin, ymax))


        old_center = src_xy.mean(axis=0)  
        new_center = new_cell_coords.mean(axis=0)  
        shift = new_center - old_center
        new_cell_coords -= shift  

        src_adata.obsm["spatial_lddmm"] = new_cell_coords.astype(np.float32)

        npz_path = os.path.join(ALIGNED_OUT, f"LDDMM_Field_{organ}_{t_src}_to_{t_tgt}.npz")
        np.savez_compressed(
            npz_path,
            phi0_final=Phi0[-1],
            det_phi1_final=detPhi1[-1],
            energies=np.asarray(energies),
            length=length,
            img_size=curr_img_size,  
            src_stage=t_src,
            tgt_stage=t_tgt,
        )

    print(f"[{organ}] LDDMM 局部配准完成")
    return adatas_aligned, jac_records



def save_aligned(adatas_aligned, actual, organ: str):
    for t, a in zip(actual, adatas_aligned):
        out_path = os.path.join(ALIGNED_OUT, f"Aligned_{organ}_{t}.h5ad")
        a.write(out_path)
        print(f"[{organ}] 保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="器官级时空对齐")
    parser.add_argument("--organ", type=str, required=True, choices=["Lung", "Brain"])
    parser.add_argument("--img-size", type=int, default=200, help="LDDMM 图像分辨率")
    parser.add_argument("--lddmm-t", type=int, default=None, help="覆盖 LDDMM 时间步数 T")
    parser.add_argument("--lddmm-k", type=int, default=None, help="覆盖 LDDMM 最大迭代次数 K")
    parser.add_argument("--lddmm-sigma", type=float, default=None, help="覆盖 LDDMM 核宽 sigma")
    parser.add_argument("--lddmm-alpha", type=float, default=None, help="覆盖 LDDMM 正则化 alpha（拉普拉斯平滑权重）")
    parser.add_argument("--lddmm-gamma", type=float, default=None, help="覆盖 LDDMM 正则化 gamma（恒等/惯性权重）")
    parser.add_argument("--lddmm-epsilon", type=float, default=None, help="覆盖 LDDMM 学习率 epsilon")
    parser.add_argument("--sdt-weight", type=float, default=None,
                        help="覆盖 mask/SDT 通道权重（默认：Lung=2.0，Brain=0.3）")
    parser.add_argument("--skip-lddmm", action="store_true", help="调试用：跳过 LDDMM")
    args = parser.parse_args()

    global organ
    ensure_dirs()
    organ = args.organ

    adatas, actual = extract_organ(organ)
    adatas = preprocess(adatas, actual)
    adatas_aligned, params, start_id = global_chain_align(adatas, actual, organ)

    if args.skip_lddmm:
        for a in adatas_aligned:
            a.obsm["spatial_lddmm"] = a.obsm["spatial_aligned"].copy()
        jac_records = []
    else:
        lddmm_params = LDDMM_PARAM_PROFILES[organ].copy()
        param_mapping = [
            ("T", "lddmm_t"),
            ("K", "lddmm_k"),
            ("sigma", "lddmm_sigma"),
            ("alpha", "lddmm_alpha"),
            ("gamma", "lddmm_gamma"),
            ("epsilon", "lddmm_epsilon"),
        ]

        is_staged = "early" in lddmm_params and "late" in lddmm_params
        for key, arg_name in param_mapping:
            val = getattr(args, arg_name, None)
            if val is not None:
                if is_staged:
                    for stage in lddmm_params:
                        lddmm_params[stage][key] = val
                else:
                    lddmm_params[key] = val

        sdt_weight = args.sdt_weight
        if sdt_weight is None:
            sdt_weight = {"Lung": 1.0, "Brain": 1.0}.get(organ, 1.0)

        if args.img_size == 200:
            img_size_map = { "Brain": 512, "Lung": 512}
            args.img_size = img_size_map.get(organ, 400)

        adatas_aligned, jac_records = lddmm_refine(
            adatas_aligned, actual, start_id, organ, args.img_size, lddmm_params,
            sdt_weight=sdt_weight,
        )

    save_aligned(adatas_aligned, actual, organ)

    print(f"[{organ}] 对齐完成！")


if __name__ == "__main__":
    main()
