# ==========================================================================
# COLAB QUICK-START GUIDE
# ==========================================================================
#
# OPTION A — Run everything with one command:
#   1. Upload spectral_fmap_experiments.py to Colab
#   2. Run: !python spectral_fmap_experiments.py
#
# OPTION B — Run cell-by-cell (recommended for first run):
#   Copy each section below into separate Colab cells.
#   This lets you inspect intermediate results and recover from disconnects.
#
# OPTION C — Quick debug run (5 min):
#   Set cfg.max_samples = 1000 in the Config to test the pipeline fast.
#
# ESTIMATED RUNTIMES (Colab T4, full Flickr30k):
#   Feature extraction:  ~20 min (one-time, cached to disk)
#   Spectral pipeline:   ~10 min
#   All experiments:     ~30 min
#   Total first run:     ~60 min
#   Subsequent runs:     ~30 min (features cached)
#
# MEMORY PROFILE:
#   Peak GPU:  ~2 GB (during DINOv2 inference)
#   Peak RAM:  ~4 GB (during eigendecomposition)
#   Disk:      ~6 GB (dataset + features + models cache)
# ==========================================================================

# %%  ===================== CELL 1: INSTALL =====================

# !pip install -q torch torchvision torchaudio
# !pip install -q transformers sentence-transformers
# !pip install -q faiss-cpu scipy scikit-learn
# !pip install -q datasets Pillow tqdm pandas tabulate seaborn matplotlib

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")


# %%  ===================== CELL 2: UPLOAD & IMPORT =====================

# Upload spectral_fmap_experiments.py to Colab, then run this cell.

import importlib, sys, os
if os.path.exists("spectral_fmap_experiments.py"):
    spec = importlib.util.spec_from_file_location("sfm", "spectral_fmap_experiments.py")
    sfm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sfm)
    Config = sfm.Config
    FeatureExtractor = sfm.FeatureExtractor
    SpectralPipeline = sfm.SpectralPipeline
    FunctionalMap = sfm.FunctionalMap
    Baselines = sfm.Baselines
    Evaluator = sfm.Evaluator
    load_flickr30k = sfm.load_flickr30k
    run_experiment_1 = sfm.run_experiment_1
    run_experiment_2 = sfm.run_experiment_2
    run_experiment_3 = sfm.run_experiment_3
    run_experiment_4 = sfm.run_experiment_4
    run_experiment_5 = sfm.run_experiment_5
    generate_figures = sfm.generate_figures
    print("Imported successfully!")
else:
    print("ERROR: Upload spectral_fmap_experiments.py first!")


# %%  ===================== CELL 3: CONFIGURE =====================

cfg = Config()

# ----- FOR QUICK DEBUG (uncomment) -----
# cfg.max_samples = 1000
# cfg.anchor_budgets = [10, 20, 50]
# cfg.spectral_k = 30
# cfg.spectral_k_max = 50

print(f"Config: max_samples={cfg.max_samples}, k={cfg.spectral_k}, device={cfg.device}")


# %%  ===================== CELL 4: LOAD DATASET =====================

images, captions, img_to_cap = load_flickr30k(cfg)
print(f"Loaded {len(images)} images, {len(captions)} captions")


# %%  ===================== CELL 5: EXTRACT FEATURES =====================

import numpy as np

img_feats_dino = FeatureExtractor.extract_dino_v2(images, cfg, "dinov2_vitb14")
txt_feats_minilm = FeatureExtractor.extract_text_features(captions, cfg, "all-MiniLM-L6-v2")

N_img = len(images)
txt_feats_per_image = np.zeros((N_img, txt_feats_minilm.shape[1]))
for i, cap_ids in enumerate(img_to_cap):
    txt_feats_per_image[i] = txt_feats_minilm[cap_ids].mean(axis=0)

print(f"Image features: {img_feats_dino.shape}")
print(f"Text features:  {txt_feats_per_image.shape}")


# %%  ===================== CELL 6: SPECTRAL PIPELINE =====================

spec_img = SpectralPipeline.full_pipeline(
    img_feats_dino, cfg.spectral_k_max, cfg.knn_k, "DINOv2 Images"
)
spec_txt = SpectralPipeline.full_pipeline(
    txt_feats_per_image, cfg.spectral_k_max, cfg.knn_k, "MiniLM Text"
)

print(f"Image spectral basis: {spec_img['Phi_k'].shape}")
print(f"Text spectral basis:  {spec_txt['Phi_k'].shape}")


# %%  ===================== CELL 7: QUICK SANITY CHECK =====================

np.random.seed(42)
anchor_idx = np.random.choice(N_img, size=20, replace=False)
k = cfg.spectral_k

C = FunctionalMap.compute_fmap_supervised(
    spec_img["Phi_k"][:, :k], spec_txt["Phi_k"][:, :k],
    spec_img["Lambda_k"][:k], spec_txt["Lambda_k"][:k],
    anchor_idx, cfg.lambda_comm, cfg.lambda_reg
)

sim = Evaluator.spectral_similarity(
    spec_img["Phi_k"][:, :k], spec_txt["Phi_k"][:, :k], C
)
N_cap = len(captions)
sim_expanded = np.zeros((N_img, N_cap))
for i, cap_ids in enumerate(img_to_cap):
    for c_id in cap_ids:
        sim_expanded[:, c_id] = sim[:, i]

res = Evaluator.compute_recall_at_k(sim_expanded, img_to_cap, [1, 5, 10], "i2t")
print(f"Quick test (20 anchors, k={k}):")
for key, val in res.items():
    print(f"  {key}: {val:.2f}%")


# %%  ===================== CELL 8: EXPERIMENT 1 =====================

results_1, cached = run_experiment_1(cfg)

import pandas as pd
df = pd.DataFrame(results_1)
print(df.to_string(index=False))


# %%  ===================== CELL 9: EXPERIMENT 2 =====================

results_2 = run_experiment_2(cfg, cached)


# %%  ===================== CELL 10: EXPERIMENT 3 =====================

results_3 = run_experiment_3(cfg, cached)


# %%  ===================== CELL 11: EXPERIMENT 4 =====================

run_experiment_4(cfg, cached)


# %%  ===================== CELL 12: EXPERIMENT 5 =====================

results_5 = run_experiment_5(cfg)


# %%  ===================== CELL 13: FIGURES =====================

generate_figures(cfg)


# %%  ===================== CELL 14: DOWNLOAD =====================

import shutil
shutil.make_archive("spectral_fmap_results", "zip", ".", "results")
shutil.make_archive("spectral_fmap_figures", "zip", ".", "figures")

try:
    from google.colab import files
    files.download("spectral_fmap_results.zip")
    files.download("spectral_fmap_figures.zip")
except ImportError:
    print("Results in ./results/ and ./figures/")
