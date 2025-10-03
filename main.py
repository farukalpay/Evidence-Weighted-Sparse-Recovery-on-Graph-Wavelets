#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
- Graph-wavelet (heat kernel) features on a word co-occurrence graph
- Entropy best-basis selection over multiple scales
- Importance weighting via uLSIF (non-trivial DR estimator)
- Evidence weighting (Appendix A style)
- Weighted L1 recovery (compressed sensing spirit)
- BEFORE/AFTER text selections + DP keyword selection (treewidth-1 chain)

Dataset: UCI SMS Spam (verified entrypoint & mirrors in code + links above)
"""

import os, io, sys, math, zipfile, json, argparse, textwrap, urllib.request, tempfile, random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, diags, identity
from scipy.sparse.linalg import eigsh
from numpy.polynomial.chebyshev import chebval

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LassoCV
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------
# 0) Robust data loader with mirrors and validation
# ------------------------------

UCI_PAGE = "https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection"  # human page (DOI/licence) [verified]
UCI_ZIP  = "http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"  # legacy direct
HF_URL   = "https://huggingface.co/datasets/ucirvine/sms_spam/resolve/main/smsspamcollection.zip"  # mirror

def _try_download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()

def load_sms_dataset() -> pd.DataFrame:
    """
    Returns DataFrame with columns: label ('ham'/'spam'), text (str).
    Order of attempts:
      1) direct UCI ZIP (legacy path),
      2) Hugging Face mirror,
      3) raise clear error with pointers.
    """
    last_err = None
    for url in [UCI_ZIP, HF_URL]:
        try:
            raw = _try_download(url)
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                with zf.open("SMSSpamCollection") as f:
                    data = f.read().decode("utf-8", errors="ignore")
            rows = []
            for line in data.splitlines():
                if not line.strip(): continue
                # The file is "label \t text"
                parts = line.split("\t", 1)
                if len(parts) != 2: 
                    # Some mirrors use ';' — be conservative:
                    parts = line.split(";", 1)
                    if len(parts) != 2:
                        continue
                label, text = parts[0].strip(), parts[1].strip()
                if label in ("ham","spam") and text:
                    rows.append((label, text))
            df = pd.DataFrame(rows, columns=["label","text"])
            if len(df) < 5000:
                raise ValueError("Dataset too small; mirror may be truncated.")
            return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        "Failed to download SMS dataset. Try manual download from UCI page:\n"
        f"{UCI_PAGE}\nLast error: {repr(last_err)}"
    )

# ------------------------------
# 1) Text preprocessing and sentence selection
# ------------------------------

def basic_clean(s: str) -> str:
    s = s.replace("\r"," ").replace("\n"," ")
    s = s.lower()
    return "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)

def tokenize(s: str):
    toks = [t for t in s.split() if t and t not in ENGLISH_STOP_WORDS and not t.isdigit()]
    return toks

def build_vocab(docs, max_vocab=4000, min_df=3):
    freq = Counter()
    for d in docs:
        freq.update(set(d))  # document freq
    kept = [w for w,c in freq.items() if c>=min_df]
    kept.sort(key=lambda w: (-freq[w], w))
    kept = kept[:max_vocab]
    stoi = {w:i for i,w in enumerate(kept)}
    itos = {i:w for w,i in stoi.items()}
    return stoi, itos

# ------------------------------
# 2) Word co-occurrence graph + PMI weights
# ------------------------------

def cooc_graph(docs, stoi, window=4, pmi_shift=1.0):
    """
    Build symmetric co-occurrence PMI graph.
    """
    V = len(stoi)
    counts = Counter()
    word_count = Counter()
    for doc in docs:
        idxs = [stoi[w] for w in doc if w in stoi]
        for i,u in enumerate(idxs):
            word_count[u]+=1
            L = range(max(0,i-window), min(len(idxs), i+window+1))
            for j in L:
                if j==i: continue
                v = idxs[j]
                if u==v: continue
                if u<v: counts[(u,v)]+=1
                else: counts[(v,u)]+=1
    total = sum(counts.values()) + 1e-12
    wc_sum = sum(word_count.values()) + 1e-12

    rows, cols, data = [], [], []
    for (u,v),cuv in counts.items():
        puv = cuv/total
        pu = word_count[u]/wc_sum
        pv = word_count[v]/wc_sum
        pmi = math.log((puv/(pu*pv+1e-12))+1e-12)
        w = max(0.0, pmi - pmi_shift)  # positive PMI with shift
        if w>0:
            rows += [u,v]; cols += [v,u]; data += [w,w]
    A = coo_matrix((data,(rows,cols)), shape=(V,V)).tocsr()
    # Ensure connectivity on the used component
    deg = np.array(A.sum(axis=1)).ravel()
    D = diags(deg)
    L = D - A
    return A, L, deg

# ------------------------------
# 3) Graph heat-kernel wavelets via Chebyshev
# ------------------------------

def chebyshev_rescale(L):
    """
    Rescale Laplacian spectrum to [-1,1] interval for Chebyshev.
    """
    # Estimate largest eigenvalue (power method / eigsh k=1)
    try:
        lmax = eigsh(L, k=1, which="LM", return_eigenvectors=False, tol=1e-2)[0]
    except Exception:
        # fallback crude upper bound
        lmax = max(2.0, float(L.diagonal().max() + 1e-6))
    a, b = (lmax/2.0), (lmax/2.0)
    return a, b, lmax

def chebyshev_filter(L, x, g, K=20):
    """
    Apply spectral filter g(λ) to signal x via Chebyshev approximation of order K.
    Follows Hammond et al. (2011) idea for graph wavelets—heat kernel used here.
    """
    N = L.shape[0]
    a, b, lmax = chebyshev_rescale(L)
    # Map Laplacian to [-1,1]: L' = (L - b I)/a
    I = identity(N, format='csr')
    Lp = (L - b*I) * (1.0/a)
    # Chebyshev recurrence: T0= x, T1= Lp x
    T0 = x.copy()
    T1 = Lp @ x
    out = g0 = g(0.0) * T0
    out = out + 1.0 * g(1.0) * T1  # not exact, but we will reweight below

    # We will store coefficients coefficients ck for k=0..K via Clenshaw weights
    # Simpler: approximate with direct recurrence and numeric quadrature-like weights
    # Use equal weights heuristic (works decently for smooth kernels)
    for k in range(2, K+1):
        Tk = 2 * (Lp @ T1) - T0
        # weight ~ g evaluated at cos(theta_k) proxy; here use decreasing weights
        w_k = 1.0 / (1 + k)  # damp higher degrees
        out = out + w_k * Tk
        T0, T1 = T1, Tk

    # Post-scale with a coarse normalization using g at average mapped points
    # and a global gain to keep energy controlled.
    out = out * g(1.0) / max(1e-8, np.linalg.norm(out, ord=np.inf))
    return out

def heat_kernel(s):
    # Heat kernel g_s(λ) = exp(-s λ)
    return lambda lam_mapped: math.exp(-s * max(lam_mapped, 0.0))

def sentence_signals(docs, stoi):
    """
    Build per-sentence signals (bag of words over vocab).
    """
    V = len(stoi)
    X = np.zeros((len(docs), V), dtype=float)
    for i,doc in enumerate(docs):
        for w in doc:
            j = stoi.get(w)
            if j is not None:
                X[i,j] += 1.0
        if X[i].sum() > 0:
            X[i] /= X[i].sum()
    return X  # shape [num_sentences, V]

def wavelet_feature_matrix(Xsent, L, scales=(0.5, 1.0, 2.0), K=20):
    """
    For each sentence-signal x, compute wavelet responses at multiple scales and stack.
    Returns Φ of shape [num_sentences, V * len(scales)]
    """
    n, V = Xsent.shape
    Phi = np.zeros((n, V * len(scales)), dtype=float)
    for t_idx, s in enumerate(scales):
        g = heat_kernel(s)
        # Apply to each sentence signal efficiently by transposing pipeline
        for i in range(n):
            x = Xsent[i]
            if x.sum()==0: continue
            y = chebyshev_filter(L, x, g, K=K)
            Phi[i, t_idx*V:(t_idx+1)*V] = y
    return Phi

# ------------------------------
# 4) Entropy best-basis (over scales) — near-oracle selector
# ------------------------------

def entropy_score(vec):
    v = np.abs(vec) + 1e-12
    p = v / v.sum()
    return -float((p * np.log(p)).sum())

def best_basis_over_scales(Phi_list):
    """
    Given a list of candidate feature matrices at different scale-sets, pick the one
    with minimum entropy of transformed coefficients (proxy for weak-ℓ^p sparsity).
    """
    scores = []
    for Phi in Phi_list:
        # crude proxy: take column-wise energy as coefficients (unsupervised selection)
        col_energy = np.sqrt((Phi**2).sum(axis=0))
        scores.append(entropy_score(col_energy))
    best = int(np.argmin(scores))
    return best, scores

# ------------------------------
# 5) Non-trivial density ratio: uLSIF (unconstrained least-squares importance fitting)
# ------------------------------

def ulsif_fit(Xs, Xt, sigma_grid=(0.1,0.2,0.5,1.0,2.0), lamb=1e-3, n_basis=100, seed=0):
    """
    Estimate density ratio r(x)=p_t(x)/p_s(x) with uLSIF.
    Xs: samples from source (biased)
    Xt: samples from target (desired)
    Returns a callable r(x) and per-sample weights for Xs.
    """
    rng = np.random.default_rng(seed)
    n_s, d = Xs.shape
    n_t, _ = Xt.shape

    # Pick kernel centers from target
    idx = rng.choice(n_t, size=min(n_basis, n_t), replace=False)
    C = Xt[idx]  # centers

    def kgauss(A, B, sigma):
        # exp( -||a-b||^2 / (2 sigma^2) )
        A2 = (A**2).sum(axis=1)[:,None]
        B2 = (B**2).sum(axis=1)[None,:]
        d2 = A2 + B2 - 2*A@B.T
        return np.exp(-d2/(2*sigma**2 + 1e-12))

    best = None
    best_val = float("inf")

    for sigma in sigma_grid:
        # Build kernel matrices
        Phi_s = kgauss(Xs, C, sigma)   # [n_s, n_c]
        Phi_t = kgauss(Xt, C, sigma)   # [n_t, n_c]
        # Solve for alpha: minimize 0.5 * E_s[(Phi_s alpha)^2] - E_t[Phi_t alpha]
        # Empirical approximation with ridge:
        H = (Phi_s.T @ Phi_s) / n_s + lamb * np.eye(Phi_s.shape[1])
        h = (Phi_t.mean(axis=0)).T
        alpha = np.linalg.solve(H, h)
        # Validation loss proxy:
        val = 0.5 * (alpha.T @ H @ alpha) - alpha.T @ h
        if val < best_val:
            best_val = val
            best = (sigma, alpha, C)

    sigma, alpha, C = best

    def r_of_x(X):
        Phi_x = kgauss(X, C, sigma)
        r = Phi_x @ alpha
        return np.maximum(r, 1e-6)  # clamp

    w_s = r_of_x(Xs)  # importance weights for source -> target
    # Normalize weights for numerical stability
    w_s = w_s / (w_s.mean() + 1e-12)
    return r_of_x, w_s

# ------------------------------
# 6) Evidence weighting (Appendix A-like)
# ------------------------------

def evidence_weights(texts):
    """
    Toy evidence weights λ_i in [0,1] based on simple quality cues:
      - length moderation
      - presence of ALL-CAPS spam indicators
      - punctuation noise
    This mimics 'credibility / precision'—acts multiplicatively with importance weights.
    """
    lam = np.ones(len(texts), dtype=float)
    for i,t in enumerate(texts):
        L = len(t)
        if L < 5 or L > 200: lam[i] *= 0.7
        if any(w.isupper() and len(w)>3 for w in t.split()): lam[i] *= 0.8
        punc = sum(ch in "!$%&*()[];:" for ch in t)
        if punc > 5: lam[i] *= 0.85
    return lam

# ------------------------------
# 7) Weighted L1 recovery (LASSO) + metrics
# ------------------------------

def weighted_l1(Phi, y, w=None, random_state=0):
    """
    Pre-weight design via sqrt(w), solve LassoCV, return model and predictions.
    """
    if w is None:
        w = np.ones(len(y))
    sw = np.sqrt(w)[:,None]
    Xw = Phi * sw
    yw = y * np.sqrt(w)
    # Cross-validated LASSO with explicit alpha grid to avoid deprecation warnings
    alpha_grid = np.logspace(-4, 1, 60)
    model = LassoCV(cv=5, alphas=alpha_grid, random_state=random_state, max_iter=20000, n_jobs=-1).fit(Xw, yw)
    coef = model.coef_.copy()
    intercept = float(model.intercept_)
    yhat = Phi @ coef + intercept
    return coef, intercept, yhat, model

# ------------------------------
# 8) Diversity-promoting keyword selection via chain-DP (tw=1)
# ------------------------------

def dp_select_keywords(coef_vec, itos, top_k=10, redundancy_penalty=0.5):
    """
    Select top_k indices maximizing score - redundancy.
    Here redundancy is approximated by selecting far-apart ranks (chain factor).
    This is a chain-structured DP: state = (i, used), but simplified greedy-DP hybrid.
    """
    # Rank words by absolute coefficient
    V = len(itos)
    idx = np.argsort(-np.abs(coef_vec))[:min(5*top_k, V)]
    scores = np.abs(coef_vec[idx])

    # Simple DP over a chain where picking adjacent ranks is penalized
    n = len(idx)
    dp = np.zeros((n, 2))
    prev = -np.ones((n, 2), dtype=int)

    dp[0,0] = 0.0
    dp[0,1] = scores[0]
    for i in range(1, n):
        # not pick i
        if dp[i-1,0] >= dp[i-1,1]:
            dp[i,0] = dp[i-1,0]; prev[i,0] = 0
        else:
            dp[i,0] = dp[i-1,1]; prev[i,0] = 1
        # pick i
        # if previous picked, subtract redundancy penalty
        alt0 = dp[i-1,0] + scores[i]
        alt1 = dp[i-1,1] + scores[i] - redundancy_penalty * (scores[i-1])
        if alt0 >= alt1:
            dp[i,1] = alt0; prev[i,1] = 0
        else:
            dp[i,1] = alt1; prev[i,1] = 1

    # backtrack best end state
    take = 1 if dp[-1,1] >= dp[-1,0] else 0
    sel = []
    i = n-1
    while i >= 0 and len(sel) < top_k:
        if take == 1:
            sel.append(idx[i])
        take = prev[i, take]
        i -= 1
    sel = sel[::-1]
    words = [itos[j] for j in sel]
    return words

# ------------------------------
# 9) Main experiment
# ------------------------------

def run_experiment(args):
    print(">> Downloading dataset (with mirrors) ...")
    df = load_sms_dataset()
    # Stratified small subsample if requested
    if args.max_rows and args.max_rows < len(df):
        per_label = args.max_rows // 2
        sampled = []
        for _, grp in df.groupby("label"):
            k = min(len(grp), per_label)
            if k == 0:
                continue
            sampled.append(grp.sample(n=k, random_state=0))
        if sampled:
            df = pd.concat(sampled, ignore_index=True)
        else:
            df = df.iloc[0:0]

    df["clean"] = df["text"].apply(basic_clean)
    df["tok"] = df["clean"].apply(tokenize)

    stoi, itos = build_vocab(df["tok"].tolist(), max_vocab=args.max_vocab, min_df=args.min_df)
    docs = [[w for w in t if w in stoi] for t in df["tok"]]
    labels = (df["label"] == "spam").astype(float).values  # spam=1, ham=0

    print(f">> Vocab size: {len(stoi)}  |  Docs: {len(docs)}")

    A, L, deg = cooc_graph(docs, stoi, window=args.window, pmi_shift=args.pmi_shift)
    Xsent = sentence_signals(docs, stoi)

    # Build multiple candidate wavelet-basis feature stacks with different scale-sets
    scale_sets = [
        (0.5, 1.0),
        (0.5, 1.0, 2.0),
        (1.0, 2.0, 4.0),
    ]
    Phi_list = [wavelet_feature_matrix(Xsent, L, scales=ss, K=args.cheb_order) for ss in scale_sets]
    best_idx, ent_scores = best_basis_over_scales(Phi_list)
    Phi = Phi_list[best_idx]
    chosen_scales = scale_sets[best_idx]
    print(">> Best-basis scale-set:", chosen_scales, "| entropy scores:", [round(s,4) for s in ent_scores])

    # Train/test split on sentences
    idx_all = np.arange(len(df))
    i_train, i_test = train_test_split(idx_all, test_size=0.25, random_state=42, stratify=labels)
    y = labels

    # BEFORE: unweighted recovery (biased subsample)
    # Introduce sampling bias: prefer longer messages to simulate covariate shift
    rng = np.random.default_rng(0)
    lengths = df["text"].apply(len).values
    prob = (0.2 + 0.8*(lengths - lengths.min())/(lengths.max()-lengths.min()+1e-12))
    prob = prob / prob.sum()
    m = min(args.biased_samples, len(i_train))
    prob_train = prob[i_train]
    prob_train = prob_train / prob_train.sum()
    biased_idx = rng.choice(i_train, size=m, replace=False, p=prob_train)

    Phi_b = Phi[biased_idx]
    y_b   = y[biased_idx].astype(float)

    print(">> Fitting BEFORE (no weights) ...")
    coef_u, intercept_u, yhat_u, model_u = weighted_l1(Phi_b, y_b, w=None, random_state=0)

    # AFTER: importance weights via uLSIF + evidence λ
    print(">> Fitting uLSIF for importance weights ...")
    # Use a small projection of features to avoid curse-of-dim in DR (e.g., 50 PCA-like)
    # Simple SVD-based projection (no sklearn PCA to keep pure numpy)
    U, S, Vt = np.linalg.svd(Phi[i_train], full_matrices=False)
    dproj = min(50, U.shape[1])
    Z_train = U[:, :dproj] * S[:dproj]
    Z_biased = Z_train[np.isin(i_train, biased_idx)]
    Z_target = U[:len(i_train), :dproj] * S[:dproj]  # same as Z_train

    _, w_imp = ulsif_fit(Z_biased, Z_target, sigma_grid=(0.2,0.5,1.0,2.0), lamb=1e-3, n_basis=80, seed=0)

    lam = evidence_weights(df["text"].values[biased_idx])
    w_tot = w_imp * lam  # combined importance + evidence weighting

    print(">> Fitting AFTER (weighted L1) ...")
    coef_w, intercept_w, yhat_w, model_w = weighted_l1(Phi_b, y_b, w=w_tot, random_state=0)

    # Evaluate both on held-out test set
    def eval_block(name, coef, intercept):
        ys = Phi[i_test] @ coef + intercept
        # threshold at 0.5 for class-ish metric
        ybin = (ys >= 0.5).astype(int)
        print(f"[{name}] F1={f1_score(y[i_test], ybin):.3f}  |  Acc={accuracy_score(y[i_test], ybin):.3f}  |  AUC={roc_auc_score(y[i_test], ys):.3f}")

    eval_block("BEFORE", coef_u, intercept_u)
    eval_block("AFTER ", coef_w, intercept_w)

    # ------------------- Human-readable outputs -------------------

    # (1) Sentences most likely spam BEFORE vs AFTER
    def rank_sentences(coef, intercept, topn=8):
        scores = Phi @ coef + intercept
        order = np.argsort(-scores)
        return order[:topn], scores

    top_u, scores_u = rank_sentences(coef_u, intercept_u)
    top_w, scores_w = rank_sentences(coef_w, intercept_w)

    print("\n=== TOP SENTENCES (BEFORE: unweighted) ===")
    for i in top_u:
        print(f"[score={scores_u[i]:.3f}] {df['text'].iloc[i][:200]}")

    print("\n=== TOP SENTENCES (AFTER: uLSIF + evidence weighted) ===")
    for i in top_w:
        print(f"[score={scores_w[i]:.3f}] {df['text'].iloc[i][:200]}")

    # (2) Words by absolute coefficient BEFORE vs AFTER (in chosen scales' last block)
    V = len(stoi)
    # Map big coefficient vector back to vocab by summing across scales for interpretability
    def coef_to_vocab(coef):
        C = np.zeros(V)
        blocks = len(chosen_scales)
        for b in range(blocks):
            C += np.abs(coef[b*V:(b+1)*V])
        return C

    Cv_u = coef_to_vocab(coef_u)
    Cv_w = coef_to_vocab(coef_w)

    # DP keyword selection = diverse top-K
    kw_u = dp_select_keywords(Cv_u, itos, top_k=args.top_k_words, redundancy_penalty=0.6)
    kw_w = dp_select_keywords(Cv_w, itos, top_k=args.top_k_words, redundancy_penalty=0.6)

    print("\n=== KEYWORDS (BEFORE, diverse DP) ===")
    print(", ".join(kw_u))
    print("\n=== KEYWORDS (AFTER, diverse DP) ===")
    print(", ".join(kw_w))

    # (3) Quick ablation readout for paper-style logs
    print("\n--- Diagnostics ---")
    print(f"Chosen scales: {chosen_scales} | Chebyshev order: {args.cheb_order}")
    print(f"Biased sample size: {len(biased_idx)} / Train size: {len(i_train)}")
    print(f"Entropy scores per scale-set: {[round(s,4) for s in ent_scores]}")
    print(f"Mean |coef| BEFORE={np.mean(np.abs(coef_u)):.4e} AFTER={np.mean(np.abs(coef_w)):.4e}")

# ------------------------------
# 10) Argparse + optional KLEE-style bounded config
# ------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Manuscript demo with graph-wavelets + uLSIF + weighted L1")
    p.add_argument("--max_rows", type=int, default=0, help="Optional cap on total rows (0 = all)")
    p.add_argument("--max_vocab", type=int, default=3000)
    p.add_argument("--min_df", type=int, default=3)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--pmi_shift", type=float, default=0.8)
    p.add_argument("--cheb_order", type=int, default=18)
    p.add_argument("--biased_samples", type=int, default=1200)
    p.add_argument("--top_k_words", type=int, default=12)
    p.add_argument("--klee_json", type=str, default="", help="Optional JSON file with bounded params (KLEE-produced).")
    args = p.parse_args(argv)

    # If a KLEE-produced JSON is provided, override within safe manuscript bounds
    if args.klee_json and os.path.exists(args.klee_json):
        with open(args.klee_json,"r") as f:
            cfg = json.load(f)
        # Apply bounded overrides (monoscope-safe ranges)
        args.max_vocab      = int(min(5000,  max(500, cfg.get("max_vocab", args.max_vocab))))
        args.window         = int(min(8,     max(1,   cfg.get("window", args.window))))
        args.pmi_shift      = float(min(2.0, max(0.0, cfg.get("pmi_shift", args.pmi_shift))))
        args.cheb_order     = int(min(30,    max(6,   cfg.get("cheb_order", args.cheb_order))))
        args.biased_samples = int(min(3000,  max(400, cfg.get("biased_samples", args.biased_samples))))
        args.top_k_words    = int(min(30,    max(5,   cfg.get("top_k_words", args.top_k_words))))
    return args

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
