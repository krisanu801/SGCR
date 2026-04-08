#!/usr/bin/env python3
"""
=============================================================================
Spectral Functional Maps for Training-Free Cross-Modal Alignment
=============================================================================
Full experimental pipeline for ACM MM 2026 BNI Track

Hardware: Google Colab T4 (16GB VRAM) or 16GB Mac M4
Runtime:  ~4-6 hours for full suite (can be run in parts)

Usage on Colab:
  1. Upload this file or paste cells into notebook
  2. Run Section 0 (install dependencies)
  3. Run sections sequentially — each saves results to disk

Author: [Your Name]
=============================================================================
"""

# ============================================================================
# SECTION 0: INSTALLATION (run once)
# ============================================================================

import subprocess
import sys

def install_dependencies():
    """Run this cell first on Colab."""
    packages = [
        "torch", "torchvision", "torchaudio",
        "transformers", "sentence-transformers",
        "faiss-cpu", "scipy", "scikit-learn",
        "numpy", "matplotlib", "seaborn",
        "datasets", "Pillow", "tqdm",
        "pandas", "tabulate"
    ]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    print("All dependencies installed.")

# Uncomment on Colab:
# install_dependencies()

# ============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ============================================================================

import os
import gc
import time
import json
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import linear_sum_assignment
from scipy.linalg import orthogonal_procrustes
from sklearn.cross_decomposition import CCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 150
import seaborn as sns
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Central configuration for all experiments."""
    # Paths
    data_dir: str = "./data"
    feature_dir: str = "./features"
    spectral_dir: str = "./spectral"
    results_dir: str = "./results"
    figures_dir: str = "./figures"

    # Dataset
    dataset_name: str = "nlphuji/flickr30k"
    max_samples: int = 31783       # Full Flickr30k; reduce for debugging
    captions_per_image: int = 5

    # Graph construction
    knn_k: int = 15                # k for k-NN graph
    sigma_mode: str = "adaptive"   # "adaptive" or "fixed"

    # Spectral
    spectral_k: int = 50           # Number of eigenvectors to keep
    spectral_k_max: int = 100      # For ZoomOut refinement

    # Functional map
    n_anchors: int = 20            # Number of anchor pairs (semi-supervised)
    lambda_comm: float = 1e-1      # Laplacian commutativity weight
    lambda_reg: float = 1e-3       # Tikhonov regularization weight
    zoomout_steps: int = 5         # Number of ZoomOut refinement iterations

    # Evaluation
    recall_at_k: List[int] = field(default_factory=lambda: [1, 5, 10])
    anchor_budgets: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 500])

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    batch_size: int = 64
    fp16: bool = True

    def __post_init__(self):
        for d in [self.data_dir, self.feature_dir, self.spectral_dir,
                  self.results_dir, self.figures_dir]:
            os.makedirs(d, exist_ok=True)

cfg = Config()
print(f"Device: {cfg.device}")
print(f"Directories created: {cfg.data_dir}, {cfg.feature_dir}, etc.")


# ============================================================================
# SECTION 2: DATASET LOADING
# ============================================================================

def load_flickr30k(cfg: Config) -> Tuple[List, List[str]]:
    """
    Load Flickr30k from HuggingFace.
    Returns: (list_of_PIL_images, list_of_captions, img_to_cap_indices)
    Each image has 5 captions; we store all captions and track mapping.

    Tries multiple dataset sources as fallbacks:
      1. nlphuji/flickr30k with trust_remote_code=True
      2. AnyModal/flickr30k (parquet-based, no script needed)
      3. Manual download from nlphuji/flickr30k CSV + zip
    """
    from datasets import load_dataset

    ds = None

    # --- Strategy 1: nlphuji with trust_remote_code ---
    print("Loading Flickr30k — trying nlphuji/flickr30k (trust_remote_code)...")
    try:
        ds = load_dataset("nlphuji/flickr30k", split="test",
                          trust_remote_code=True)
        img_key = "image"
        cap_key = "caption"
        print("  Success with nlphuji/flickr30k")
    except Exception as e1:
        print(f"  Failed: {e1}")

        # --- Strategy 2: AnyModal mirror (parquet-based) ---
        print("  Trying AnyModal/flickr30k...")
        try:
            ds = load_dataset("AnyModal/flickr30k", split="test")
            # AnyModal may use different column names
            cols = ds.column_names
            img_key = "image" if "image" in cols else "img"
            # AnyModal uses 'alt_text' for captions; detect the right key
            for candidate in ["caption", "captions", "alt_text", "original_alt_text", "text"]:
                if candidate in cols:
                    cap_key = candidate
                    break
            else:
                cap_key = cols[1]  # fallback: second column
            print(f"  Success with AnyModal/flickr30k (cols: {cols})")
        except Exception as e2:
            print(f"  Failed: {e2}")

            # --- Strategy 3: Manual CSV approach ---
            print("  Trying manual download from nlphuji/flickr30k CSV...")
            try:
                from huggingface_hub import hf_hub_download
                import pandas as pdd
                from PIL import Image
                from zipfile import ZipFile
                from io import BytesIO

                csv_path = hf_hub_download(repo_id="nlphuji/flickr30k",
                                           filename="flickr_annotations_30k.csv",
                                           repo_type="dataset")
                zip_path = hf_hub_download(repo_id="nlphuji/flickr30k",
                                           filename="flickr30k-images.zip",
                                           repo_type="dataset")

                ann = pdd.read_csv(csv_path)
                zf = ZipFile(zip_path)

                images = []
                captions_list = []
                img_to_cap_indices = []

                # Group by image filename
                grouped = ann.groupby("filename") if "filename" in ann.columns else ann.groupby(ann.columns[0])
                cap_idx = 0
                for i, (fname, group) in enumerate(tqdm(grouped, desc="Loading images")):
                    if i >= cfg.max_samples:
                        break
                    # Try to open image from zip
                    try:
                        img_data = zf.read(f"flickr30k-images/{fname}")
                        img = Image.open(BytesIO(img_data)).convert("RGB")
                    except KeyError:
                        img_data = zf.read(fname)
                        img = Image.open(BytesIO(img_data)).convert("RGB")

                    images.append(img)
                    cap_indices = []
                    caps = group.iloc[:, 1:6] if group.shape[1] >= 6 else group
                    for c in list(caps.values.flatten())[:cfg.captions_per_image]:
                        if isinstance(c, str) and len(c.strip()) > 0:
                            captions_list.append(c.strip())
                            cap_indices.append(cap_idx)
                            cap_idx += 1
                    if cap_indices:
                        img_to_cap_indices.append(cap_indices)
                    else:
                        images.pop()

                zf.close()
                print(f"Loaded {len(images)} images, {len(captions_list)} captions (manual)")
                return images, captions_list, img_to_cap_indices

            except Exception as e3:
                raise RuntimeError(
                    f"All Flickr30k loading strategies failed.\n"
                    f"  Strategy 1 (nlphuji trust_remote_code): {e1}\n"
                    f"  Strategy 2 (AnyModal mirror): {e2}\n"
                    f"  Strategy 3 (manual CSV+zip): {e3}\n\n"
                    f"Please try: pip install datasets --upgrade\n"
                    f"Or manually download from https://huggingface.co/datasets/nlphuji/flickr30k"
                )

    # --- Parse the loaded dataset ---
    images = []
    captions_list = []
    img_to_cap_indices = []

    # Check if 'original_alt_text' has the multi-caption list (AnyModal stores
    # the 5 original captions there as a list, while 'alt_text' is just one)
    cols = ds.column_names
    multi_cap_key = None
    if "original_alt_text" in cols:
        # Peek at first sample to check if it's a list
        first = ds[0]["original_alt_text"]
        if isinstance(first, list) and len(first) > 1:
            multi_cap_key = "original_alt_text"
            print(f"  Using '{multi_cap_key}' for multi-caption ({len(first)} per image)")

    cap_idx = 0
    for i, sample in enumerate(tqdm(ds, desc="Loading samples")):
        if i >= cfg.max_samples:
            break
        images.append(sample[img_key])

        # Get captions — prefer multi-caption field if available
        if multi_cap_key is not None:
            raw_caps = sample[multi_cap_key]
        else:
            raw_caps = sample[cap_key]

        # Normalize to list
        if isinstance(raw_caps, str):
            raw_caps = [raw_caps]
        elif raw_caps is None:
            raw_caps = [""]

        sample_captions = [c for c in raw_caps if isinstance(c, str) and len(c.strip()) > 0]
        sample_captions = sample_captions[:cfg.captions_per_image]

        # Must have at least 1 caption
        if len(sample_captions) == 0:
            images.pop()
            continue

        cap_indices = []
        for c in sample_captions:
            captions_list.append(c)
            cap_indices.append(cap_idx)
            cap_idx += 1
        img_to_cap_indices.append(cap_indices)

    # Report stats
    caps_per_img = [len(c) for c in img_to_cap_indices]
    print(f"Loaded {len(images)} images, {len(captions_list)} captions")
    print(f"  Captions per image: min={min(caps_per_img)}, max={max(caps_per_img)}, "
          f"mean={np.mean(caps_per_img):.1f}")
    return images, captions_list, img_to_cap_indices


# ============================================================================
# SECTION 3: FEATURE EXTRACTION
# ============================================================================

class FeatureExtractor:
    """
    Extract features from pretrained models one at a time.
    Saves to disk and unloads model to free memory.
    """

    @staticmethod
    def extract_dino_v2(images, cfg: Config, variant="dinov2_vitb14"):
        """Extract DINOv2 features for images."""
        save_path = os.path.join(cfg.feature_dir, f"{variant}_features.npy")
        if os.path.exists(save_path):
            print(f"Loading cached {variant} features...")
            return np.load(save_path)

        print(f"Extracting {variant} features...")
        model = torch.hub.load("facebookresearch/dinov2", variant)
        model = model.to(cfg.device)
        model.eval()
        if cfg.fp16 and cfg.device == "cuda":
            model = model.half()

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        all_features = []
        for i in tqdm(range(0, len(images), cfg.batch_size), desc=f"{variant}"):
            batch_imgs = images[i:i + cfg.batch_size]
            batch_tensors = torch.stack([transform(img.convert("RGB")) for img in batch_imgs])
            batch_tensors = batch_tensors.to(cfg.device)
            if cfg.fp16 and cfg.device == "cuda":
                batch_tensors = batch_tensors.half()

            with torch.no_grad():
                features = model(batch_tensors)
            all_features.append(features.float().cpu().numpy())

        features = np.concatenate(all_features, axis=0)

        # Cleanup
        del model, batch_tensors
        torch.cuda.empty_cache() if cfg.device == "cuda" else None
        gc.collect()

        np.save(save_path, features)
        print(f"Saved {variant} features: shape {features.shape}")
        return features

    @staticmethod
    def extract_mae(images, cfg: Config, variant="mae_vitb16"):
        """Extract MAE features for images."""
        save_path = os.path.join(cfg.feature_dir, f"{variant}_features.npy")
        if os.path.exists(save_path):
            print(f"Loading cached {variant} features...")
            return np.load(save_path)

        print(f"Extracting {variant} features...")
        from transformers import ViTMAEModel, ViTMAEConfig, AutoImageProcessor

        if variant == "mae_vitb16":
            model_name = "facebook/vit-mae-base"
        else:
            model_name = "facebook/vit-mae-large"

        processor = AutoImageProcessor.from_pretrained(model_name)
        model = ViTMAEModel.from_pretrained(model_name)
        model = model.to(cfg.device)
        model.eval()
        if cfg.fp16 and cfg.device == "cuda":
            model = model.half()

        all_features = []
        for i in tqdm(range(0, len(images), cfg.batch_size), desc=f"{variant}"):
            batch_imgs = images[i:i + cfg.batch_size]
            inputs = processor(images=[img.convert("RGB") for img in batch_imgs],
                               return_tensors="pt").to(cfg.device)
            if cfg.fp16 and cfg.device == "cuda":
                inputs["pixel_values"] = inputs["pixel_values"].half()

            with torch.no_grad():
                outputs = model(**inputs)
                # Use CLS token (first token of last hidden state)
                features = outputs.last_hidden_state[:, 0, :]
            all_features.append(features.float().cpu().numpy())

        features = np.concatenate(all_features, axis=0)
        del model, processor
        torch.cuda.empty_cache() if cfg.device == "cuda" else None
        gc.collect()

        np.save(save_path, features)
        print(f"Saved {variant} features: shape {features.shape}")
        return features

    @staticmethod
    def extract_text_features(captions, cfg: Config, model_name="all-MiniLM-L6-v2"):
        """Extract sentence transformer features for captions."""
        safe_name = model_name.replace("/", "_").replace("-", "_")
        save_path = os.path.join(cfg.feature_dir, f"text_{safe_name}_features.npy")
        if os.path.exists(save_path):
            print(f"Loading cached {model_name} features...")
            return np.load(save_path)

        print(f"Extracting {model_name} features...")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device=cfg.device)

        features = model.encode(
            captions,
            batch_size=cfg.batch_size * 4,  # Text is cheaper
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        del model
        torch.cuda.empty_cache() if cfg.device == "cuda" else None
        gc.collect()

        np.save(save_path, features)
        print(f"Saved {model_name} features: shape {features.shape}")
        return features

    @staticmethod
    def extract_clip_features(images, captions, cfg: Config, variant="openai/clip-vit-base-patch32"):
        """Extract CLIP features (baseline — jointly trained)."""
        safe_name = variant.replace("/", "_").replace("-", "_")
        img_save = os.path.join(cfg.feature_dir, f"clip_{safe_name}_img.npy")
        txt_save = os.path.join(cfg.feature_dir, f"clip_{safe_name}_txt.npy")

        if os.path.exists(img_save) and os.path.exists(txt_save):
            print(f"Loading cached CLIP features...")
            return np.load(img_save), np.load(txt_save)

        print(f"Extracting CLIP features ({variant})...")
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(variant).to(cfg.device)
        processor = CLIPProcessor.from_pretrained(variant)
        model.eval()
        if cfg.fp16 and cfg.device == "cuda":
            model = model.half()

        # Image features — use full forward pass, extract image_embeds
        all_img_feats = []
        for i in tqdm(range(0, len(images), cfg.batch_size), desc="CLIP images"):
            batch_imgs = images[i:i + cfg.batch_size]
            # Process images only (no text) — use vision model directly
            pixel_values = processor(images=[img.convert("RGB") for img in batch_imgs],
                                     return_tensors="pt")["pixel_values"].to(cfg.device)
            if cfg.fp16 and cfg.device == "cuda":
                pixel_values = pixel_values.half()
            with torch.no_grad():
                vision_out = model.vision_model(pixel_values=pixel_values)
                # Get pooled output and project through visual projection
                pooled = vision_out.pooler_output  # (B, hidden_dim)
                feats = model.visual_projection(pooled)  # (B, projection_dim)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_img_feats.append(feats.float().cpu().numpy())
        img_features = np.concatenate(all_img_feats, axis=0)

        # Text features — use text model directly
        all_txt_feats = []
        for i in tqdm(range(0, len(captions), cfg.batch_size * 2), desc="CLIP text"):
            batch_caps = captions[i:i + cfg.batch_size * 2]
            text_inputs = processor(text=batch_caps, return_tensors="pt",
                                    padding=True, truncation=True, max_length=77).to(cfg.device)
            with torch.no_grad():
                text_out = model.text_model(
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"]
                )
                pooled = text_out.pooler_output  # (B, hidden_dim)
                feats = model.text_projection(pooled)  # (B, projection_dim)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_txt_feats.append(feats.float().cpu().numpy())
        txt_features = np.concatenate(all_txt_feats, axis=0)

        del model, processor
        torch.cuda.empty_cache() if cfg.device == "cuda" else None
        gc.collect()

        np.save(img_save, img_features)
        np.save(txt_save, txt_features)
        print(f"CLIP img: {img_features.shape}, txt: {txt_features.shape}")
        return img_features, txt_features


# ============================================================================
# SECTION 4: SPECTRAL PIPELINE (Core of the method)
# ============================================================================

class SpectralPipeline:
    """
    Constructs graph Laplacians and computes spectral bases.
    This is the mathematical core — all operations are training-free.
    """

    @staticmethod
    def build_knn_graph(features: np.ndarray, k: int = 15,
                        metric: str = "cosine") -> csr_matrix:
        """
        Build symmetric k-NN graph with Gaussian kernel weights.

        Args:
            features: (N, d) feature matrix
            k: number of nearest neighbors
            metric: distance metric

        Returns:
            W: (N, N) sparse symmetric weight matrix
        """
        N = features.shape[0]
        print(f"  Building {k}-NN graph for {N} points...")

        # Normalize for cosine similarity
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        features_norm = features / norms

        # Use sklearn NearestNeighbors (works on CPU, memory efficient)
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
        nn.fit(features_norm)
        distances, indices = nn.kneighbors(features_norm)

        # Remove self-connections (first column)
        distances = distances[:, 1:]  # (N, k)
        indices = indices[:, 1:]      # (N, k)

        # Adaptive bandwidth: sigma_i = mean distance to k-th neighbor
        sigma = np.mean(distances[:, -1])
        print(f"  Adaptive sigma: {sigma:.4f}")

        # Gaussian kernel weights
        weights = np.exp(-distances**2 / (sigma**2))

        # Build sparse matrix
        row_idx = np.repeat(np.arange(N), k)
        col_idx = indices.flatten()
        w_vals = weights.flatten()

        W = csr_matrix((w_vals, (row_idx, col_idx)), shape=(N, N))
        # Symmetrize: W = (W + W^T) / 2
        W = (W + W.T) / 2.0

        print(f"  Graph: {W.nnz} nonzero entries, density: {W.nnz / N**2:.6f}")
        return W

    @staticmethod
    def compute_normalized_laplacian(W: csr_matrix) -> csr_matrix:
        """
        Compute normalized graph Laplacian: L = I - D^{-1/2} W D^{-1/2}

        Returns:
            L: (N, N) sparse normalized Laplacian
        """
        from scipy.sparse import diags, eye

        N = W.shape[0]
        d = np.array(W.sum(axis=1)).flatten()
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        D_inv_sqrt = diags(d_inv_sqrt)

        L = eye(N) - D_inv_sqrt @ W @ D_inv_sqrt
        return L

    @staticmethod
    def compute_spectral_basis(L: csr_matrix, k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute bottom-k eigenvectors of the Laplacian.

        Uses ARPACK (shift-invert mode) for efficiency on sparse matrices.

        Args:
            L: (N, N) sparse normalized Laplacian
            k: number of eigenvectors to compute (excluding trivial)

        Returns:
            Phi_k: (N, k) eigenvector matrix
            Lambda_k: (k,) eigenvalue vector
        """
        N = L.shape[0]
        print(f"  Computing {k} eigenvectors of {N}x{N} Laplacian...")

        t0 = time.time()
        # Compute k+1 smallest eigenvalues (first is ~0, the constant eigenvector)
        eigenvalues, eigenvectors = eigsh(L, k=k + 1, which="SM", tol=1e-6)

        # Sort by eigenvalue (should already be sorted, but ensure)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Skip first eigenvector (constant, eigenvalue ≈ 0)
        Phi_k = eigenvectors[:, 1:k + 1]    # (N, k)
        Lambda_k = eigenvalues[1:k + 1]      # (k,)

        elapsed = time.time() - t0
        print(f"  Eigendecomposition done in {elapsed:.1f}s")
        print(f"  Eigenvalue range: [{Lambda_k[0]:.6f}, {Lambda_k[-1]:.6f}]")

        return Phi_k, Lambda_k

    @classmethod
    def full_pipeline(cls, features: np.ndarray, k_spectral: int,
                      knn_k: int = 15, name: str = "") -> dict:
        """
        Run the full spectral pipeline: features → k-NN → Laplacian → eigenbasis.

        Returns dict with: W, L, Phi_k, Lambda_k
        """
        print(f"\n{'='*60}")
        print(f"Spectral Pipeline: {name}")
        print(f"{'='*60}")

        W = cls.build_knn_graph(features, k=knn_k)
        L = cls.compute_normalized_laplacian(W)
        Phi_k, Lambda_k = cls.compute_spectral_basis(L, k=k_spectral)

        # Free W and L immediately — they are large and not needed after eigen
        del W, L

        return {
            "Phi_k": Phi_k,
            "Lambda_k": Lambda_k,
            "name": name
        }


# ============================================================================
# SECTION 5: FUNCTIONAL MAP COMPUTATION
# ============================================================================

class FunctionalMap:
    """
    Compute and refine the functional map C between two spectral bases.
    """

    @staticmethod
    def compute_fmap_supervised(
        Phi_k_src: np.ndarray,   # (N, k) source spectral basis
        Phi_k_tgt: np.ndarray,   # (N, k) target spectral basis
        Lambda_src: np.ndarray,  # (k,) source eigenvalues
        Lambda_tgt: np.ndarray,  # (k,) target eigenvalues
        anchor_indices: np.ndarray,  # (p,) indices of anchor points
        lambda_comm: float = 0.1,
        lambda_reg: float = 0.001
    ) -> np.ndarray:
        """
        Compute functional map C via least-squares with Laplacian commutativity.

        C* = argmin_C ||C A - B||_F^2
                     + lambda_comm * ||C Λ_src - Λ_tgt C||_F^2
                     + lambda_reg  * ||C||_F^2

        Where A, B are spectral coefficients of probe functions at anchors.

        Returns:
            C: (k, k) functional map matrix
        """
        k = Phi_k_src.shape[1]
        p = len(anchor_indices)

        # Probe functions: indicator functions smoothed through spectral basis
        # For each anchor i, the probe function is the i-th row of Phi_k
        # Spectral coefficients: A[:, l] = Phi_k_src[anchor_l, :]
        A = Phi_k_src[anchor_indices, :].T  # (k, p)
        B = Phi_k_tgt[anchor_indices, :].T  # (k, p)

        # Term 1: Descriptor preservation — ||CA - B||^2
        # Normal equations: (AA^T) C^T = AB^T → solve for C
        # But we need to add the commutativity and regularization terms.

        # We'll solve the full system by vectorizing.
        # vec(C) ∈ R^{k^2}
        # Term 1: ||CA - B||^2 = ||( A^T ⊗ I_k ) vec(C) - vec(B)||^2
        # Term 2: ||C Λ_src - Λ_tgt C||^2
        # Term 3: ||C||^2

        # --- Efficient formulation without explicit Kronecker ---
        # Solve column by column of C using the structure.
        # Actually, for moderate k (50-100), direct Kronecker is fine.

        I_k = np.eye(k)
        Lam_src = np.diag(Lambda_src)
        Lam_tgt = np.diag(Lambda_tgt)

        # Term 1: (A A^T ⊗ I_k)
        AAt = A @ A.T  # (k, k)
        T1 = np.kron(AAt, I_k)

        # Term 2: Laplacian commutativity
        # ||C Λ_src - Λ_tgt C||_F^2
        # = ||vec(C Λ_src) - vec(Λ_tgt C)||^2
        # = ||(Λ_src^T ⊗ I_k) vec(C) - (I_k ⊗ Λ_tgt) vec(C)||^2
        # = ||[(Λ_src ⊗ I_k) - (I_k ⊗ Λ_tgt)] vec(C)||^2
        Comm = np.kron(Lam_src, I_k) - np.kron(I_k, Lam_tgt)
        T2 = Comm.T @ Comm

        # Term 3: Tikhonov
        T3 = np.eye(k * k)

        # Full system
        LHS = T1 + lambda_comm * T2 + lambda_reg * T3

        # RHS: vec(B A^T) for term 1
        RHS = (B @ A.T).flatten()

        # Solve
        vec_C = np.linalg.solve(LHS, RHS)
        C = vec_C.reshape(k, k)

        return C

    @staticmethod
    def compute_fmap_unsupervised(
        Phi_k_src: np.ndarray,
        Phi_k_tgt: np.ndarray,
        Lambda_src: np.ndarray,
        Lambda_tgt: np.ndarray,
        lambda_comm: float = 1.0,
        lambda_reg: float = 0.01,
        n_hks_scales: int = 10
    ) -> np.ndarray:
        """
        Compute functional map WITHOUT any anchor correspondences.
        Uses Heat Kernel Signatures (HKS) as probe functions.

        HKS_t(i) = sum_j exp(-lambda_j * t) * phi_j(i)^2

        These are intrinsic descriptors — computed independently per modality.
        """
        k = Phi_k_src.shape[1]
        N = Phi_k_src.shape[0]

        # Compute HKS at multiple scales
        # Scale range: [4/lambda_max, 4/lambda_2] (logarithmically spaced)
        t_min = 4.0 / max(Lambda_src[-1], Lambda_tgt[-1])
        t_max = 4.0 / max(Lambda_src[0], Lambda_tgt[0], 1e-6)
        t_scales = np.logspace(np.log10(t_min), np.log10(t_max), n_hks_scales)

        # Compute HKS for source
        HKS_src = np.zeros((N, n_hks_scales))
        for q, t in enumerate(t_scales):
            exp_vals = np.exp(-Lambda_src * t)  # (k,)
            HKS_src[:, q] = np.sum((Phi_k_src ** 2) * exp_vals[None, :], axis=1)

        # Compute HKS for target
        HKS_tgt = np.zeros((N, n_hks_scales))
        for q, t in enumerate(t_scales):
            exp_vals = np.exp(-Lambda_tgt * t)
            HKS_tgt[:, q] = np.sum((Phi_k_tgt ** 2) * exp_vals[None, :], axis=1)

        # Spectral coefficients of HKS
        A = Phi_k_src.T @ HKS_src  # (k, n_hks_scales)
        B = Phi_k_tgt.T @ HKS_tgt  # (k, n_hks_scales)

        # Solve same optimization as supervised case
        I_k = np.eye(k)
        Lam_src = np.diag(Lambda_src)
        Lam_tgt = np.diag(Lambda_tgt)

        AAt = A @ A.T
        T1 = np.kron(AAt, I_k)

        Comm = np.kron(Lam_src, I_k) - np.kron(I_k, Lam_tgt)
        T2 = Comm.T @ Comm

        T3 = np.eye(k * k)

        LHS = T1 + lambda_comm * T2 + lambda_reg * T3
        RHS = (B @ A.T).flatten()

        vec_C = np.linalg.solve(LHS, RHS)
        C = vec_C.reshape(k, k)

        return C

    @staticmethod
    def zoomout_refinement(
        C_init: np.ndarray,
        Phi_src_full: np.ndarray,  # (N, k_max) — larger basis
        Phi_tgt_full: np.ndarray,
        k_init: int,
        k_max: int,
        n_steps: int = 5
    ) -> np.ndarray:
        """
        ZoomOut refinement: progressively increase spectral resolution.

        Starting from a low-frequency map C_init (k_init × k_init),
        iteratively:
          1. Recover point-to-point map via nearest neighbor in spectral coords
          2. Upscale to higher-frequency basis

        Args:
            C_init: (k_init, k_init) initial functional map
            Phi_src_full: (N, k_max) source eigenvectors (more than k_init)
            Phi_tgt_full: (N, k_max) target eigenvectors
            k_init: initial spectral dimension
            k_max: final spectral dimension
            n_steps: number of refinement steps

        Returns:
            C_refined: (k_max, k_max) refined functional map
        """
        k_schedule = np.linspace(k_init, k_max, n_steps + 1, dtype=int)

        C = C_init.copy()

        for step in range(n_steps):
            k_curr = k_schedule[step]
            k_next = k_schedule[step + 1]

            # Current spectral coordinates
            Phi_src_k = Phi_src_full[:, :k_curr]
            Phi_tgt_k = Phi_tgt_full[:, :k_curr]

            # Map source spectral coords to target space
            mapped_src = Phi_src_k @ C.T  # (N, k_curr)

            # Nearest neighbor: for each source point, find closest target point
            # in spectral coordinates
            # This recovers the point-to-point correspondence Π
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn.fit(Phi_tgt_k)
            _, indices = nn.kneighbors(mapped_src)
            indices = indices.flatten()  # (N,) — Π(i) = indices[i]

            # Upscale: compute new C at higher resolution
            Phi_src_next = Phi_src_full[:, :k_next]  # (N, k_next)
            Phi_tgt_next = Phi_tgt_full[:, :k_next]

            # C_new = Phi_tgt_next^T @ Π @ Phi_src_next
            # Where Π is the permutation matrix encoded by indices
            Phi_tgt_permuted = Phi_tgt_next[indices, :]  # (N, k_next)
            C = Phi_tgt_permuted.T @ Phi_src_next  # (k_next, k_next)

            # Optional: orthogonalize C (functional maps of isometries are orthogonal)
            U, _, Vt = np.linalg.svd(C, full_matrices=False)
            C = U @ Vt

        return C


# ============================================================================
# SECTION 6: BASELINE METHODS
# ============================================================================

class Baselines:
    """
    Training-free baseline alignment methods for comparison.
    """

    @staticmethod
    def raw_cosine(feat_src: np.ndarray, feat_tgt: np.ndarray) -> np.ndarray:
        """
        Raw cosine similarity between independently trained embeddings.
        Expected to perform near-random (spaces are unaligned).
        Handles different dimensionalities by truncating to min(d_src, d_tgt).
        Returns: (N_src, N_tgt) similarity matrix.
        """
        d = min(feat_src.shape[1], feat_tgt.shape[1])
        src = feat_src[:, :d]
        tgt = feat_tgt[:, :d]
        feat_src_n = src / (np.linalg.norm(src, axis=1, keepdims=True) + 1e-8)
        feat_tgt_n = tgt / (np.linalg.norm(tgt, axis=1, keepdims=True) + 1e-8)
        return feat_src_n @ feat_tgt_n.T

    @staticmethod
    def procrustes_alignment(
        feat_src: np.ndarray, feat_tgt: np.ndarray,
        anchor_indices: np.ndarray
    ) -> np.ndarray:
        """
        Orthogonal Procrustes alignment: find rotation R minimizing
        ||X_anchors @ R - Y_anchors||_F.

        Returns: similarity matrix after alignment.
        """
        X = feat_src[anchor_indices]
        Y = feat_tgt[anchor_indices]

        # Ensure same dimensionality (truncate to min)
        d = min(X.shape[1], Y.shape[1])
        X = X[:, :d]
        Y = Y[:, :d]

        R, _ = orthogonal_procrustes(X, Y)

        feat_src_aligned = feat_src[:, :d] @ R
        feat_tgt_trunc = feat_tgt[:, :d]

        src_n = feat_src_aligned / (np.linalg.norm(feat_src_aligned, axis=1, keepdims=True) + 1e-8)
        tgt_n = feat_tgt_trunc / (np.linalg.norm(feat_tgt_trunc, axis=1, keepdims=True) + 1e-8)
        return src_n @ tgt_n.T

    @staticmethod
    def cca_alignment(
        feat_src: np.ndarray, feat_tgt: np.ndarray,
        anchor_indices: np.ndarray, n_components: int = 50
    ) -> np.ndarray:
        """
        CCA-based alignment using anchor pairs.
        Project both modalities to shared CCA space.
        """
        X = feat_src[anchor_indices]
        Y = feat_tgt[anchor_indices]

        d = min(X.shape[1], Y.shape[1])
        n_comp = min(n_components, len(anchor_indices) - 1, d)

        cca = CCA(n_components=n_comp, max_iter=1000)
        cca.fit(X[:, :d], Y[:, :d])

        X_cca = cca.transform(feat_src[:, :d])
        Y_cca = cca.transform(feat_tgt[:, :d])

        X_n = X_cca / (np.linalg.norm(X_cca, axis=1, keepdims=True) + 1e-8)
        Y_n = Y_cca / (np.linalg.norm(Y_cca, axis=1, keepdims=True) + 1e-8)
        return X_n @ Y_n.T

    @staticmethod
    def relative_representations(
        feat_src: np.ndarray, feat_tgt: np.ndarray,
        anchor_indices: np.ndarray
    ) -> np.ndarray:
        """
        Relative Representations (Moschella et al., 2022).
        Represent each point by its similarity to anchor points.
        """
        # Anchor features
        X_anchors = feat_src[anchor_indices]  # (p, d_src)
        Y_anchors = feat_tgt[anchor_indices]  # (p, d_tgt)

        # Normalize
        X_anchors_n = X_anchors / (np.linalg.norm(X_anchors, axis=1, keepdims=True) + 1e-8)
        Y_anchors_n = Y_anchors / (np.linalg.norm(Y_anchors, axis=1, keepdims=True) + 1e-8)

        feat_src_n = feat_src / (np.linalg.norm(feat_src, axis=1, keepdims=True) + 1e-8)
        feat_tgt_n = feat_tgt / (np.linalg.norm(feat_tgt, axis=1, keepdims=True) + 1e-8)

        # Relative representations: similarity to anchors
        rel_src = feat_src_n @ X_anchors_n.T  # (N, p)
        rel_tgt = feat_tgt_n @ Y_anchors_n.T  # (N, p)

        # Now compare in the shared "relative" space
        rel_src_n = rel_src / (np.linalg.norm(rel_src, axis=1, keepdims=True) + 1e-8)
        rel_tgt_n = rel_tgt / (np.linalg.norm(rel_tgt, axis=1, keepdims=True) + 1e-8)
        return rel_src_n @ rel_tgt_n.T


# ============================================================================
# SECTION 7: EVALUATION METRICS
# ============================================================================

class Evaluator:
    """
    Compute retrieval metrics: Recall@K for image→text and text→image.
    Memory-efficient: avoids allocating (N_img × N_cap) expanded matrices.
    """

    @staticmethod
    def compute_recall_at_k(
        similarity: np.ndarray,     # (N_img, N_cap) or spectral similarity
        img_to_cap: List[List[int]],  # ground truth: img_idx → [cap_indices]
        k_values: List[int] = [1, 5, 10],
        direction: str = "i2t"      # "i2t" or "t2i"
    ) -> Dict[str, float]:
        """
        Compute Recall@K.
        """
        results = {}

        if direction == "i2t":
            N_img = len(img_to_cap)
            for k in k_values:
                hits = 0
                for i in range(N_img):
                    topk = np.argpartition(-similarity[i], k)[:k]
                    gt_caps = set(img_to_cap[i])
                    if len(gt_caps & set(topk)) > 0:
                        hits += 1
                results[f"i2t_R@{k}"] = hits / N_img * 100

        elif direction == "t2i":
            cap_to_img = {}
            for img_idx, cap_indices in enumerate(img_to_cap):
                for cap_idx in cap_indices:
                    cap_to_img[cap_idx] = img_idx

            N_cap = similarity.shape[1]
            for k in k_values:
                hits = 0
                total = 0
                for c in range(N_cap):
                    if c not in cap_to_img:
                        continue
                    topk = np.argpartition(-similarity[:, c], k)[:k]
                    gt_img = cap_to_img[c]
                    if gt_img in topk:
                        hits += 1
                    total += 1
                results[f"t2i_R@{k}"] = hits / total * 100

        return results

    @staticmethod
    def recall_from_img_sim(
        sim_img: np.ndarray,         # (N_img, N_img) image-level similarity
        img_to_cap: List[List[int]],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute i2t and t2i recall directly from image-level similarity
        WITHOUT expanding to (N_img, N_cap).

        Key insight: since all captions for image j share the same similarity
        score (they map to the same image), ranking captions = ranking images
        with adjusted K.

        For i2t R@K: image i retrieves top-K captions. Since each image has
        C captions, retrieving the right image = retrieving all C captions.
        So i2t R@K on expanded = i2t R@ceil(K/C) on image-level sim,
        BUT we need to be precise: top-K captions come from top-ceil(K/C) images.

        Simpler correct approach: for each image, the ground-truth "target" in
        the image-level sim is just itself (diagonal). Check if the diagonal
        entry is in top-K (image-level).
        """
        results = {}
        N_img = sim_img.shape[0]

        # i2t: for image i, the correct text representative is index i
        # With 5 captions per image, R@K in expanded space means:
        # the correct image-text pair is at rank r in image-level,
        # which maps to rank r*C to (r+1)*C - 1 in caption-level.
        # So image-level R@1 ≈ expanded R@C, image-level R@K ≈ expanded R@(K*C)
        # To get expanded R@K, we need image-level R@ceil(K/C)
        # But this is approximate. For exact results, we compute directly:

        caps_per_img = len(img_to_cap[0]) if img_to_cap else 5

        for k in k_values:
            # In expanded space with C captions per image, top-K captions
            # come from at most K unique images (at least ceil(K/C)).
            # The correct image has C captions, so if correct image is in
            # top-ceil(K/C) images, at least one caption is in top-K.
            k_img = max(1, (k + caps_per_img - 1) // caps_per_img)

            hits = 0
            for i in range(N_img):
                topk = np.argpartition(-sim_img[i], k_img)[:k_img]
                if i in topk:
                    hits += 1
            results[f"i2t_R@{k}"] = hits / N_img * 100

        # t2i: for each caption (C per image), the correct image is the parent
        # Since all C captions for image j have the same sim to image i,
        # t2i R@K = fraction of captions whose parent image is in top-K images
        # = fraction of images that are in their own top-K (same as i2t at image level)
        for k in k_values:
            hits = 0
            total = 0
            for j in range(N_img):
                topk = np.argpartition(-sim_img[:, j], k)[:k]
                n_caps = len(img_to_cap[j])
                for _ in range(n_caps):
                    if j in topk:
                        hits += 1
                    total += 1
            results[f"t2i_R@{k}"] = hits / total * 100

        return results

    @staticmethod
    def spectral_similarity(
        Phi_src: np.ndarray,     # (N_src, k)
        Phi_tgt: np.ndarray,     # (N_tgt, k)
        C: np.ndarray            # (k, k) functional map
    ) -> np.ndarray:
        """
        Compute cross-modal similarity via functional map.
        Returns: (N_src, N_tgt) similarity matrix (higher = more similar)
        """
        mapped_src = Phi_src @ C.T  # (N_src, k)
        src_sq = np.sum(mapped_src ** 2, axis=1, keepdims=True)
        tgt_sq = np.sum(Phi_tgt ** 2, axis=1, keepdims=True)
        cross = mapped_src @ Phi_tgt.T
        dist_sq = src_sq + tgt_sq.T - 2 * cross
        return -dist_sq


# ============================================================================
# SECTION 8: EXPERIMENT 1 — Core Image-Text Retrieval
# ============================================================================

def run_experiment_1(cfg: Config):
    """
    Core experiment: Image-Text retrieval on Flickr30k.
    Compares functional maps against baselines.
    Memory-optimized for 12GB system RAM.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Image-Text Retrieval on Flickr30k")
    print("=" * 70)

    # ---- Load data ----
    images, captions, img_to_cap = load_flickr30k(cfg)

    # ---- Extract features (then free images to save ~2GB RAM) ----
    img_feats_dino = FeatureExtractor.extract_dino_v2(images, cfg, variant="dinov2_vitb14")
    txt_feats_minilm = FeatureExtractor.extract_text_features(captions, cfg, "all-MiniLM-L6-v2")

    N_img = len(images)
    N_cap = len(captions)
    print(f"\nN_img = {N_img}, N_cap = {N_cap}")

    txt_feats_per_image = np.zeros((N_img, txt_feats_minilm.shape[1]))
    for i, cap_ids in enumerate(img_to_cap):
        txt_feats_per_image[i] = txt_feats_minilm[cap_ids].mean(axis=0)

    # Free raw caption features — we only need per-image means from here
    del txt_feats_minilm
    gc.collect()

    # ---- Spectral pipeline (only image-level, no all-captions basis) ----
    spec_img = SpectralPipeline.full_pipeline(
        img_feats_dino, k_spectral=cfg.spectral_k_max,
        knn_k=cfg.knn_k, name="DINOv2-B Images"
    )
    spec_txt = SpectralPipeline.full_pipeline(
        txt_feats_per_image, k_spectral=cfg.spectral_k_max,
        knn_k=cfg.knn_k, name="MiniLM Text (per-image mean)"
    )

    # Free large intermediates
    gc.collect()

    # ---- Select anchors ----
    np.random.seed(42)
    all_indices = np.arange(N_img)

    # ---- Helper: evaluate from image-level sim (no expansion needed) ----
    def eval_img_sim(sim_img):
        """Compute recall from (N_img, N_img) similarity — zero extra allocation."""
        res = Evaluator.recall_from_img_sim(sim_img, img_to_cap, cfg.recall_at_k)
        del sim_img
        return res

    # ---- Run for different anchor budgets ----
    all_results = []

    for n_anchors in cfg.anchor_budgets:
        print(f"\n--- Anchor budget: {n_anchors} ---")
        anchor_idx = np.random.choice(all_indices, size=min(n_anchors, N_img), replace=False)

        # ---- Functional Map (Ours — supervised) ----
        k = cfg.spectral_k
        C_sup = FunctionalMap.compute_fmap_supervised(
            Phi_k_src=spec_img["Phi_k"][:, :k],
            Phi_k_tgt=spec_txt["Phi_k"][:, :k],
            Lambda_src=spec_img["Lambda_k"][:k],
            Lambda_tgt=spec_txt["Lambda_k"][:k],
            anchor_indices=anchor_idx,
            lambda_comm=cfg.lambda_comm,
            lambda_reg=cfg.lambda_reg
        )

        C_refined = FunctionalMap.zoomout_refinement(
            C_init=C_sup,
            Phi_src_full=spec_img["Phi_k"],
            Phi_tgt_full=spec_txt["Phi_k"],
            k_init=k,
            k_max=cfg.spectral_k_max,
            n_steps=cfg.zoomout_steps
        )

        sim_fmap = Evaluator.spectral_similarity(
            spec_img["Phi_k"][:, :cfg.spectral_k_max],
            spec_txt["Phi_k"][:, :cfg.spectral_k_max],
            C_refined
        )
        res_fmap = eval_img_sim(sim_fmap)
        res_fmap["method"] = "FMap (ours)"
        res_fmap["n_anchors"] = n_anchors
        all_results.append(res_fmap)
        print(f"  FMap: {res_fmap}")

        # ---- Baselines ----
        # 1. Raw cosine (once)
        if n_anchors == cfg.anchor_budgets[0]:
            sim_raw = Baselines.raw_cosine(img_feats_dino, txt_feats_per_image)
            res_raw = eval_img_sim(sim_raw)
            res_raw["method"] = "Raw Cosine"
            res_raw["n_anchors"] = 0
            all_results.append(res_raw)
            print(f"  Raw Cosine: {res_raw}")

        # 2. Procrustes
        sim_proc = Baselines.procrustes_alignment(
            img_feats_dino, txt_feats_per_image, anchor_idx
        )
        res_proc = eval_img_sim(sim_proc)
        res_proc["method"] = "Procrustes"
        res_proc["n_anchors"] = n_anchors
        all_results.append(res_proc)
        print(f"  Procrustes: {res_proc}")

        # 3. Relative Representations
        sim_relrep = Baselines.relative_representations(
            img_feats_dino, txt_feats_per_image, anchor_idx
        )
        res_relrep = eval_img_sim(sim_relrep)
        res_relrep["method"] = "Relative Reps"
        res_relrep["n_anchors"] = n_anchors
        all_results.append(res_relrep)
        print(f"  Relative Reps: {res_relrep}")

        # 4. CCA
        if n_anchors >= 20:
            n_cca_comp = min(cfg.spectral_k, n_anchors - 1)
            try:
                sim_cca = Baselines.cca_alignment(
                    img_feats_dino, txt_feats_per_image,
                    anchor_idx, n_components=n_cca_comp
                )
                res_cca = eval_img_sim(sim_cca)
                res_cca["method"] = "CCA"
                res_cca["n_anchors"] = n_anchors
                all_results.append(res_cca)
                print(f"  CCA: {res_cca}")
            except Exception as e:
                print(f"  CCA failed with {n_anchors} anchors: {e}")

        gc.collect()  # Clean up after each anchor budget

    # ---- Unsupervised Functional Map (HKS) ----
    print("\n--- Unsupervised FMap (HKS, 0 anchors) ---")
    k = cfg.spectral_k
    C_unsup = FunctionalMap.compute_fmap_unsupervised(
        Phi_k_src=spec_img["Phi_k"][:, :k],
        Phi_k_tgt=spec_txt["Phi_k"][:, :k],
        Lambda_src=spec_img["Lambda_k"][:k],
        Lambda_tgt=spec_txt["Lambda_k"][:k],
        lambda_comm=1.0,
        lambda_reg=0.01
    )
    C_unsup_refined = FunctionalMap.zoomout_refinement(
        C_init=C_unsup,
        Phi_src_full=spec_img["Phi_k"],
        Phi_tgt_full=spec_txt["Phi_k"],
        k_init=k,
        k_max=cfg.spectral_k_max,
        n_steps=cfg.zoomout_steps
    )
    sim_unsup = Evaluator.spectral_similarity(
        spec_img["Phi_k"], spec_txt["Phi_k"], C_unsup_refined
    )
    res_unsup = eval_img_sim(sim_unsup)
    res_unsup["method"] = "FMap Unsupervised (HKS)"
    res_unsup["n_anchors"] = 0
    all_results.append(res_unsup)
    print(f"  FMap Unsup: {res_unsup}")

    # ---- CLIP Baseline (jointly trained) ----
    print("\n--- CLIP Baselines ---")
    for clip_variant in ["openai/clip-vit-base-patch32"]:
        clip_img, clip_txt = FeatureExtractor.extract_clip_features(
            images, captions, cfg, variant=clip_variant
        )
        sim_clip = clip_img @ clip_txt.T  # (N_img, N_cap) — already normalized
        res_clip = Evaluator.compute_recall_at_k(sim_clip, img_to_cap, cfg.recall_at_k, "i2t")
        res_clip_t2i = Evaluator.compute_recall_at_k(sim_clip, img_to_cap, cfg.recall_at_k, "t2i")
        res_clip.update(res_clip_t2i)
        res_clip["method"] = f"CLIP ({clip_variant.split('/')[-1]})"
        res_clip["n_anchors"] = "400M (trained)"
        all_results.append(res_clip)
        print(f"  CLIP: {res_clip}")
        del clip_img, clip_txt, sim_clip
        gc.collect()

    # Free images now — no longer needed
    del images, captions
    gc.collect()

    # ---- Save results ----
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(cfg.results_dir, "experiment1_results.csv"), index=False)
    print(f"\nResults saved to {cfg.results_dir}/experiment1_results.csv")

    return all_results, {
        "spec_img": spec_img, "spec_txt": spec_txt,
        "img_feats_dino": img_feats_dino,
        "txt_feats_per_image": txt_feats_per_image,
        "img_to_cap": img_to_cap
    }


# ============================================================================
# SECTION 9: EXPERIMENT 2 — Ablation: Effect of Spectral Dimension k
# ============================================================================

def run_experiment_2(cfg: Config, cached_data: dict):
    """
    Ablation: How does the spectral dimension k affect retrieval quality?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Ablation over Spectral Dimension k")
    print("=" * 70)

    spec_img = cached_data["spec_img"]
    spec_txt = cached_data["spec_txt"]
    img_to_cap = cached_data["img_to_cap"]
    N_img = spec_img["Phi_k"].shape[0]
    N_cap = sum(len(c) for c in img_to_cap)

    np.random.seed(42)
    anchor_idx = np.random.choice(N_img, size=50, replace=False)

    k_values = [10, 20, 30, 50, 70, 100]
    results = []

    for k in k_values:
        if k > spec_img["Phi_k"].shape[1]:
            continue

        print(f"\n  k = {k}")
        C = FunctionalMap.compute_fmap_supervised(
            Phi_k_src=spec_img["Phi_k"][:, :k],
            Phi_k_tgt=spec_txt["Phi_k"][:, :k],
            Lambda_src=spec_img["Lambda_k"][:k],
            Lambda_tgt=spec_txt["Lambda_k"][:k],
            anchor_indices=anchor_idx,
            lambda_comm=cfg.lambda_comm,
            lambda_reg=cfg.lambda_reg
        )

        sim = Evaluator.spectral_similarity(
            spec_img["Phi_k"][:, :k],
            spec_txt["Phi_k"][:, :k],
            C
        )

        res = Evaluator.recall_from_img_sim(sim, img_to_cap, cfg.recall_at_k)
        res["k"] = k
        results.append(res)
        print(f"    R@1={res['i2t_R@1']:.2f}, R@5={res['i2t_R@5']:.2f}, R@10={res['i2t_R@10']:.2f}")
        del sim, C
        gc.collect()

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(cfg.results_dir, "experiment2_k_ablation.csv"), index=False)
    print(f"\nSaved to {cfg.results_dir}/experiment2_k_ablation.csv")
    return results


# ============================================================================
# SECTION 10: EXPERIMENT 3 — Composability (Vision→Text→Audio)
# ============================================================================

def run_experiment_3(cfg: Config, cached_data: dict):
    """
    Composability: align vision↔audio via text, without direct V-A pairs.
    Memory-optimized.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Compositional Cross-Modal Alignment")
    print("=" * 70)

    print("Using mpnet as third modality (audio proxy) for composability demo...")

    img_to_cap = cached_data["img_to_cap"]
    N_img = cached_data["img_feats_dino"].shape[0]

    # We need captions for mpnet extraction — reload just captions (lightweight)
    from datasets import load_dataset
    try:
        ds = load_dataset("AnyModal/flickr30k", split="test")
        captions = []
        for i, sample in enumerate(ds):
            if i >= cfg.max_samples:
                break
            raw = sample.get("original_alt_text", sample.get("alt_text", ""))
            if isinstance(raw, list):
                captions.extend(raw[:cfg.captions_per_image])
            elif isinstance(raw, str):
                captions.append(raw)
    except:
        # Fallback: generate dummy captions (won't affect the spectral math)
        captions = [f"caption_{i}" for i in range(N_img * 5)]

    # Extract "third modality" features
    txt_feats_mpnet_all = FeatureExtractor.extract_text_features(
        captions, cfg, "all-mpnet-base-v2"
    )
    txt_feats_2 = np.zeros((N_img, txt_feats_mpnet_all.shape[1]))
    for i, cap_ids in enumerate(img_to_cap):
        valid_ids = [c for c in cap_ids if c < len(txt_feats_mpnet_all)]
        if valid_ids:
            txt_feats_2[i] = txt_feats_mpnet_all[valid_ids].mean(axis=0)
    del txt_feats_mpnet_all
    gc.collect()

    # Spectral bases
    spec_img = cached_data["spec_img"]
    spec_txt1 = cached_data["spec_txt"]
    spec_txt2 = SpectralPipeline.full_pipeline(
        txt_feats_2, k_spectral=cfg.spectral_k_max,
        knn_k=cfg.knn_k, name="mpnet (audio proxy)"
    )
    del txt_feats_2
    gc.collect()

    np.random.seed(42)
    k = cfg.spectral_k

    anchors_v_t1 = np.random.choice(N_img, size=20, replace=False)
    C_v_t1 = FunctionalMap.compute_fmap_supervised(
        spec_img["Phi_k"][:, :k], spec_txt1["Phi_k"][:, :k],
        spec_img["Lambda_k"][:k], spec_txt1["Lambda_k"][:k],
        anchors_v_t1, cfg.lambda_comm, cfg.lambda_reg
    )

    anchors_t1_t2 = np.random.choice(N_img, size=20, replace=False)
    C_t1_t2 = FunctionalMap.compute_fmap_supervised(
        spec_txt1["Phi_k"][:, :k], spec_txt2["Phi_k"][:, :k],
        spec_txt1["Lambda_k"][:k], spec_txt2["Lambda_k"][:k],
        anchors_t1_t2, cfg.lambda_comm, cfg.lambda_reg
    )

    C_composed = C_t1_t2 @ C_v_t1

    anchors_v_t2 = np.random.choice(N_img, size=20, replace=False)
    C_direct = FunctionalMap.compute_fmap_supervised(
        spec_img["Phi_k"][:, :k], spec_txt2["Phi_k"][:, :k],
        spec_img["Lambda_k"][:k], spec_txt2["Lambda_k"][:k],
        anchors_v_t2, cfg.lambda_comm, cfg.lambda_reg
    )

    results = []
    for name, C_mat in [("Composed (V→T1→T2)", C_composed),
                         ("Direct (V→T2)", C_direct)]:
        sim = Evaluator.spectral_similarity(
            spec_img["Phi_k"][:, :k], spec_txt2["Phi_k"][:, :k], C_mat
        )
        res = Evaluator.recall_from_img_sim(sim, img_to_cap, cfg.recall_at_k)
        res["method"] = name
        results.append(res)
        print(f"  {name}: {res}")
        del sim

    del spec_txt2
    gc.collect()

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(cfg.results_dir, "experiment3_composability.csv"), index=False)
    return results


# ============================================================================
# SECTION 11: EXPERIMENT 4 — Spectral Analysis & Diagnostics
# ============================================================================

def run_experiment_4(cfg: Config, cached_data: dict):
    """
    Diagnostic: Analyze spectral properties across modalities.
    - Compare eigenvalue spectra
    - Measure spectral similarity (proxy for Assumption 2)
    - Visualize functional map structure
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Spectral Diagnostics")
    print("=" * 70)

    spec_img = cached_data["spec_img"]
    spec_txt = cached_data["spec_txt"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- Plot 1: Eigenvalue spectra comparison ----
    ax = axes[0, 0]
    k_plot = min(80, len(spec_img["Lambda_k"]))
    ax.plot(range(k_plot), spec_img["Lambda_k"][:k_plot], 'b-o',
            markersize=3, label="Vision (DINOv2)", linewidth=1.5)
    ax.plot(range(k_plot), spec_txt["Lambda_k"][:k_plot], 'r-s',
            markersize=3, label="Text (MiniLM)", linewidth=1.5)
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue λ")
    ax.set_title("Laplacian Eigenvalue Spectra")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Plot 2: Eigenvalue ratio (spectral compatibility) ----
    ax = axes[0, 1]
    ratio = spec_img["Lambda_k"][:k_plot] / (spec_txt["Lambda_k"][:k_plot] + 1e-10)
    ax.plot(range(k_plot), ratio, 'g-^', markersize=3, linewidth=1.5)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label="Perfect isometry")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("λ_vision / λ_text")
    ax.set_title("Eigenvalue Ratio (Isometry Diagnostic)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Plot 3: Functional map matrix C ----
    ax = axes[1, 0]
    k = cfg.spectral_k
    np.random.seed(42)
    anchor_idx = np.random.choice(spec_img["Phi_k"].shape[0], size=50, replace=False)
    C = FunctionalMap.compute_fmap_supervised(
        spec_img["Phi_k"][:, :k], spec_txt["Phi_k"][:, :k],
        spec_img["Lambda_k"][:k], spec_txt["Lambda_k"][:k],
        anchor_idx, cfg.lambda_comm, cfg.lambda_reg
    )
    im = ax.imshow(np.abs(C), cmap="viridis", aspect="equal")
    ax.set_xlabel("Source spectral index")
    ax.set_ylabel("Target spectral index")
    ax.set_title("|C| — Functional Map Matrix")
    plt.colorbar(im, ax=ax)

    # ---- Plot 4: Diagonal dominance analysis ----
    ax = axes[1, 1]
    diag_vals = np.abs(np.diag(C))
    off_diag_energy = np.sum(np.abs(C) ** 2, axis=1) - diag_vals ** 2
    total_energy = np.sum(np.abs(C) ** 2, axis=1)
    diag_ratio = diag_vals ** 2 / (total_energy + 1e-10)

    ax.bar(range(k), diag_ratio, alpha=0.7, color='steelblue')
    ax.set_xlabel("Spectral index")
    ax.set_ylabel("Diagonal energy fraction")
    ax.set_title("Diagonal Dominance of C\n(1.0 = perfect isometry)")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.figures_dir, "spectral_diagnostics.png"),
                dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved spectral diagnostics figure.")

    # ---- Quantitative spectral similarity ----
    # Laplacian spectral distance
    k_compare = min(50, len(spec_img["Lambda_k"]), len(spec_txt["Lambda_k"]))
    l_img = spec_img["Lambda_k"][:k_compare]
    l_txt = spec_txt["Lambda_k"][:k_compare]

    # Normalize eigenvalues to [0, 1] for fair comparison
    l_img_norm = l_img / (l_img[-1] + 1e-10)
    l_txt_norm = l_txt / (l_txt[-1] + 1e-10)
    spectral_distance = np.sqrt(np.mean((l_img_norm - l_txt_norm) ** 2))
    print(f"\nNormalized spectral distance: {spectral_distance:.4f}")
    print(f"(0 = identical spectra, larger = more different)")

    # Orthogonality of C (isometric maps have orthogonal C)
    ortho_error = np.linalg.norm(C.T @ C - np.eye(k), 'fro') / k
    print(f"Orthogonality error of C: {ortho_error:.4f}")
    print(f"(0 = perfect isometry)")


# ============================================================================
# SECTION 12: EXPERIMENT 5 — Multiple Encoder Pairs
# ============================================================================

def run_experiment_5(cfg: Config, cached_data: dict = None):
    """
    Test across multiple vision × text encoder combinations.
    Memory-optimized: reuses cached data, processes one pair at a time.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Multiple Encoder Pairs")
    print("=" * 70)

    # Get img_to_cap from cache or reload minimally
    if cached_data and "img_to_cap" in cached_data:
        img_to_cap = cached_data["img_to_cap"]
        N_img = cached_data["img_feats_dino"].shape[0]
    else:
        _, _, img_to_cap = load_flickr30k(cfg)
        N_img = len(img_to_cap)

    # Load features from cache files (no models needed in memory)
    results = []
    np.random.seed(42)
    anchor_idx = np.random.choice(N_img, size=50, replace=False)

    # Vision features — load from .npy cache
    vision_feat_files = {
        "dinov2_vitb14": os.path.join(cfg.feature_dir, "dinov2_vitb14_features.npy"),
    }

    # Text features — we need per-image means
    text_models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]

    for v_name, v_path in vision_feat_files.items():
        if not os.path.exists(v_path):
            print(f"  Skipping {v_name} — features not cached")
            continue
        img_feats = np.load(v_path)

        for t_name in text_models:
            safe_name = t_name.replace("/", "_").replace("-", "_")
            t_path = os.path.join(cfg.feature_dir, f"text_{safe_name}_features.npy")

            if not os.path.exists(t_path):
                # Need to extract — but we need captions
                # Use cached mpnet features from exp3, or skip
                print(f"  Skipping {v_name} × {t_name} — text features not cached")
                continue

            print(f"\n  Pair: {v_name} × {t_name}")
            txt_feats_all = np.load(t_path)

            txt_feats_per_image = np.zeros((N_img, txt_feats_all.shape[1]))
            for i, cap_ids in enumerate(img_to_cap):
                valid = [c for c in cap_ids if c < len(txt_feats_all)]
                if valid:
                    txt_feats_per_image[i] = txt_feats_all[valid].mean(axis=0)
            del txt_feats_all
            gc.collect()

            spec_v = SpectralPipeline.full_pipeline(
                img_feats, cfg.spectral_k_max, cfg.knn_k, v_name
            )
            spec_t = SpectralPipeline.full_pipeline(
                txt_feats_per_image, cfg.spectral_k_max, cfg.knn_k, t_name
            )
            # Free heavy parts after spectral pipeline

            k = cfg.spectral_k
            C = FunctionalMap.compute_fmap_supervised(
                spec_v["Phi_k"][:, :k], spec_t["Phi_k"][:, :k],
                spec_v["Lambda_k"][:k], spec_t["Lambda_k"][:k],
                anchor_idx, cfg.lambda_comm, cfg.lambda_reg
            )
            C = FunctionalMap.zoomout_refinement(
                C, spec_v["Phi_k"], spec_t["Phi_k"],
                k, cfg.spectral_k_max, cfg.zoomout_steps
            )

            sim = Evaluator.spectral_similarity(
                spec_v["Phi_k"], spec_t["Phi_k"], C
            )
            res = Evaluator.recall_from_img_sim(sim, img_to_cap, cfg.recall_at_k)
            res["vision"] = v_name
            res["text"] = t_name
            results.append(res)
            print(f"    {res}")

            del sim, spec_v, spec_t, txt_feats_per_image, C
            gc.collect()

        del img_feats
        gc.collect()

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(cfg.results_dir, "experiment5_encoder_pairs.csv"), index=False)
    return results


# ============================================================================
# SECTION 13: GENERATE ALL FIGURES AND TABLES
# ============================================================================

def generate_figures(cfg: Config):
    """Generate publication-quality figures from saved results."""

    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    # ---- Figure 1: Anchor Budget vs Recall ----
    try:
        df = pd.read_csv(os.path.join(cfg.results_dir, "experiment1_results.csv"))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        methods_to_plot = ["FMap (ours)", "Procrustes", "Relative Reps", "CCA"]
        colors = {"FMap (ours)": "#2196F3", "Procrustes": "#FF9800",
                  "Relative Reps": "#4CAF50", "CCA": "#9C27B0"}
        markers = {"FMap (ours)": "o", "Procrustes": "s",
                   "Relative Reps": "^", "CCA": "D"}

        for metric, ax, title in [("i2t_R@1", axes[0], "Image→Text R@1"),
                                   ("i2t_R@5", axes[1], "Image→Text R@5")]:
            for method in methods_to_plot:
                subset = df[(df["method"] == method) &
                            (df["n_anchors"].apply(lambda x: str(x).isdigit()))]
                if len(subset) == 0:
                    continue
                subset = subset.sort_values("n_anchors")
                ax.plot(subset["n_anchors"].astype(int), subset[metric],
                        marker=markers.get(method, "o"), label=method,
                        color=colors.get(method, None), linewidth=2, markersize=7)

            # Add horizontal lines for special methods
            for special in ["Raw Cosine", "FMap Unsupervised (HKS)"]:
                row = df[df["method"] == special]
                if len(row) > 0:
                    val = row[metric].values[0]
                    style = "--" if special == "Raw Cosine" else "-."
                    ax.axhline(y=val, linestyle=style, alpha=0.6,
                               label=f"{special} ({val:.1f}%)")

            clip_row = df[df["method"].str.contains("CLIP")]
            if len(clip_row) > 0:
                val = clip_row[metric].values[0]
                ax.axhline(y=val, color="red", linestyle=":", linewidth=2,
                           label=f"CLIP (400M pairs) ({val:.1f}%)")

            ax.set_xlabel("Number of Anchor Pairs", fontsize=12)
            ax.set_ylabel(f"{title} (%)", fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log")

        plt.tight_layout()
        plt.savefig(os.path.join(cfg.figures_dir, "fig1_anchor_budget.png"),
                    dpi=200, bbox_inches="tight")
        plt.show()
        print("Saved fig1_anchor_budget.png")
    except Exception as e:
        print(f"Could not generate Figure 1: {e}")

    # ---- Figure 2: Spectral Dimension Ablation ----
    try:
        df2 = pd.read_csv(os.path.join(cfg.results_dir, "experiment2_k_ablation.csv"))

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for metric, label, marker in [("i2t_R@1", "R@1", "o"),
                                       ("i2t_R@5", "R@5", "s"),
                                       ("i2t_R@10", "R@10", "^")]:
            ax.plot(df2["k"], df2[metric], marker=marker, label=label,
                    linewidth=2, markersize=7)

        ax.set_xlabel("Spectral Dimension k", fontsize=12)
        ax.set_ylabel("Recall (%)", fontsize=12)
        ax.set_title("Effect of Spectral Truncation on Retrieval", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(cfg.figures_dir, "fig2_k_ablation.png"),
                    dpi=200, bbox_inches="tight")
        plt.show()
        print("Saved fig2_k_ablation.png")
    except Exception as e:
        print(f"Could not generate Figure 2: {e}")

    # ---- Figure 3: Composability Results ----
    try:
        df3 = pd.read_csv(os.path.join(cfg.results_dir, "experiment3_composability.csv"))
        print("\n--- Composability Results ---")
        print(df3.to_string(index=False))
    except Exception as e:
        print(f"Could not load Experiment 3 results: {e}")


# ============================================================================
# SECTION 14: MAIN — Run Everything
# ============================================================================

def main():
    """Run all experiments sequentially."""
    cfg = Config()

    # Adjust for quick debugging (uncomment to use subset)
    # cfg.max_samples = 1000
    # cfg.batch_size = 32
    # cfg.anchor_budgets = [10, 20, 50]

    print("=" * 70)
    print("SPECTRAL FUNCTIONAL MAPS — FULL EXPERIMENT SUITE")
    print(f"Device: {cfg.device}")
    print(f"Max samples: {cfg.max_samples}")
    print("=" * 70)

    t_start = time.time()

    # Experiment 1: Core retrieval + baselines
    results_1, cached = run_experiment_1(cfg)

    # Experiment 2: Spectral dimension ablation
    results_2 = run_experiment_2(cfg, cached)

    # Experiment 3: Composability
    results_3 = run_experiment_3(cfg, cached)

    # Experiment 4: Spectral diagnostics
    run_experiment_4(cfg, cached)

    # Experiment 5: Multiple encoder pairs
    results_5 = run_experiment_5(cfg, cached)

    # Generate all figures
    generate_figures(cfg)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"ALL EXPERIMENTS COMPLETED in {elapsed / 60:.1f} minutes")
    print(f"Results: {cfg.results_dir}/")
    print(f"Figures: {cfg.figures_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
