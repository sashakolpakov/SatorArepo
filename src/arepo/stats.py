"""
Statistical utilities for the 4D mantissa engine and MoE experiments.
"""

import logging
import numpy as np
from scipy import stats

# Suppress "Token indices sequence length is longer than the specified maximum"
# warnings from transformers — we intentionally tokenize without truncation and
# chunk manually.
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def extract_mantissas(embedding_tensor):
    """Extract log10 mantissas: log10(|x|) mod 1 in [0, 1).

    Parameters
    ----------
    embedding_tensor : array-like or torch.Tensor
        Raw embedding values (any shape, will be flattened).

    Returns
    -------
    np.ndarray
        Fractional parts of log10(|x|) for all nonzero finite values.
    """
    if hasattr(embedding_tensor, 'numpy'):
        flat_values = embedding_tensor.flatten().numpy()
    else:
        flat_values = np.array(embedding_tensor).flatten()

    flat_values = flat_values[np.isfinite(flat_values)]
    abs_values = np.abs(flat_values)
    abs_values = abs_values[abs_values > 0]

    if len(abs_values) == 0:
        return np.array([])

    mantissas = np.log10(abs_values) % 1.0
    return mantissas


def compute_mantissa_multi_stats(mantissas):
    """Compute 5 uniformity/Benford statistics from mantissa array.

    Parameters
    ----------
    mantissas : np.ndarray
        Mantissa values in [0, 1).

    Returns
    -------
    dict
        Keys: ks, cvm, ad, benford_mse, benford_kl.
        Values are floats (or np.nan if mantissas is empty).
    """
    nan_result = {k: np.nan for k in ('ks', 'cvm', 'ad', 'benford_mse', 'benford_kl')}

    if len(mantissas) == 0:
        return nan_result

    # Mantissa uniformity tests
    ks_stat, _ = stats.kstest(mantissas, 'uniform')
    cvm_result = stats.cramervonmises(mantissas, 'uniform')
    cvm_stat = cvm_result.statistic

    # Anderson-Darling against Uniform(0,1)
    clean = mantissas[np.isfinite(mantissas)]
    if len(clean) < 2:
        ad_stat = np.nan
    else:
        sorted_m = np.sort(clean)
        n = len(sorted_m)
        # Clip to (0, 1) open interval — avoids log(0) from exact powers of 10
        sorted_m = np.clip(sorted_m, 1e-15, 1 - 1e-15)
        i = np.arange(1, n + 1)
        ad_stat = -n - np.mean((2 * i - 1) * (np.log(sorted_m) + np.log(1 - sorted_m[::-1])))
        if not np.isfinite(ad_stat):
            ad_stat = np.nan

    # Benford first-digit stats (from raw values that generated the mantissas)
    # Reconstruct leading digits from mantissas: 10^mantissa gives significand
    significands = 10 ** mantissas
    leading_digits = np.floor(significands).astype(int)
    leading_digits = leading_digits[(leading_digits >= 1) & (leading_digits <= 9)]

    if len(leading_digits) == 0:
        return dict(ks=ks_stat, cvm=cvm_stat, ad=ad_stat,
                    benford_mse=np.nan, benford_kl=np.nan)

    # Empirical first-digit distribution
    counts = np.zeros(9)
    for d in range(1, 10):
        counts[d - 1] = np.sum(leading_digits == d)
    p_hat = counts / counts.sum()

    # Benford expected
    p_benford = np.log10(1 + 1.0 / np.arange(1, 10))

    # MSE
    benford_mse = float(np.mean((p_hat - p_benford) ** 2))

    # KL divergence (with smoothing to avoid log(0))
    p_smooth = np.maximum(p_hat, 1e-10)
    benford_kl = float(np.sum(p_smooth * np.log(p_smooth / p_benford)))

    return dict(ks=ks_stat, cvm=cvm_stat, ad=float(ad_stat),
                benford_mse=benford_mse, benford_kl=benford_kl)


def get_embeddings(model, input_ids, attention_mask, is_generative):
    """Extract embeddings from a model for a single chunk.

    Parameters
    ----------
    model : torch.nn.Module
        A HuggingFace model (already on device, in eval mode).
    input_ids : torch.Tensor, shape (1, seq_len)
        Token IDs (already on device).
    attention_mask : torch.Tensor, shape (1, seq_len)
        Attention mask (already on device).
    is_generative : bool
        If True, extract token embeddings directly (generative model).
        If False, use last hidden state (masked model).

    Returns
    -------
    torch.Tensor
        Embedding tensor (on CPU).
    """
    import torch

    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    with torch.no_grad():
        if is_generative:
            if hasattr(model, 'wte'):
                embeddings = model.wte(input_ids)
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
                embeddings = model.transformer.wte(input_ids)
            elif hasattr(model, 'embed_in'):
                embeddings = model.embed_in(input_ids)
            elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                embeddings = model.model.embed_tokens(input_ids)
            else:
                outputs = model(**inputs, output_hidden_states=True)
                embeddings = outputs.hidden_states[-1]
        else:
            outputs = model(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1]
    return embeddings.cpu()


def _chunk_ranges(total_tokens, max_length):
    """Generate (start, end) ranges for overlapping chunks.

    Parameters
    ----------
    total_tokens : int
    max_length : int

    Yields
    ------
    (int, int)
        Start and end indices for each chunk.
    """
    stride = max_length // 2
    start = 0
    while start < total_tokens:
        end = min(start + max_length, total_tokens)
        # Last chunk: slide back for a full-sized window
        if end - start < max_length and total_tokens >= max_length:
            start = total_tokens - max_length
            end = total_tokens
        yield start, end
        if end == total_tokens:
            break
        start += stride


def compute_ks_chunked(text, tokenizer, model, device, is_generative,
                       max_length=None):
    """Compute KS statistic with chunked aggregation for long texts.

    For texts that fit within ``max_length`` tokens, behaves identically to the
    original single-pass logic.  For longer texts, splits into overlapping
    chunks (50 % overlap), computes a KS stat per chunk, and returns the mean.

    Parameters
    ----------
    text : str
    tokenizer : transformers.PreTrainedTokenizer
    model : torch.nn.Module
        Already on *device* and in eval mode.
    device : torch.device
    is_generative : bool
    max_length : int or None
        Defaults to 1024 for generative, 512 for masked models.

    Returns
    -------
    float
        KS statistic vs Uniform(0,1), or np.nan on failure.
    """
    import torch

    if max_length is None:
        max_length = 1024 if is_generative else 512

    # Use encode() to avoid "sequence length is longer than max" warnings
    all_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True))
    total_tokens = len(all_ids)

    if total_tokens <= max_length:
        chunk_ids = all_ids.unsqueeze(0).to(device)
        chunk_mask = torch.ones_like(chunk_ids)
        embeddings = get_embeddings(model, chunk_ids, chunk_mask, is_generative)
        mantissas = extract_mantissas(embeddings)
        if len(mantissas) == 0:
            return np.nan
        ks_stat, _ = stats.kstest(mantissas, 'uniform')
        return ks_stat

    # Long text — chunked aggregation
    chunk_ks = []
    for start, end in _chunk_ranges(total_tokens, max_length):
        chunk_ids = all_ids[start:end].unsqueeze(0).to(device)
        chunk_mask = torch.ones_like(chunk_ids)
        embeddings = get_embeddings(model, chunk_ids, chunk_mask, is_generative)
        mantissas = extract_mantissas(embeddings)
        if len(mantissas) > 0:
            ks_stat, _ = stats.kstest(mantissas, 'uniform')
            chunk_ks.append(ks_stat)

    return float(np.mean(chunk_ks)) if chunk_ks else np.nan


def compute_multi_stats_chunked(text, tokenizer, model, device, is_generative,
                                max_length=None):
    """Compute multi-statistics with chunked aggregation for long texts.

    Like :func:`compute_ks_chunked` but returns all 5 mantissa statistics.
    Per-chunk stats are averaged independently.

    Parameters
    ----------
    text, tokenizer, model, device, is_generative, max_length
        Same as :func:`compute_ks_chunked`.

    Returns
    -------
    dict
        Keys: ks, cvm, ad, benford_mse, benford_kl.
    """
    import torch

    if max_length is None:
        max_length = 1024 if is_generative else 512

    # Use encode() to avoid "sequence length is longer than max" warnings
    all_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True))
    total_tokens = len(all_ids)

    if total_tokens <= max_length:
        chunk_ids = all_ids.unsqueeze(0).to(device)
        chunk_mask = torch.ones_like(chunk_ids)
        embeddings = get_embeddings(model, chunk_ids, chunk_mask, is_generative)
        mantissas = extract_mantissas(embeddings)
        return compute_mantissa_multi_stats(mantissas)

    # Long text — chunked aggregation, average per-stat
    chunk_stats = []
    for start, end in _chunk_ranges(total_tokens, max_length):
        chunk_ids = all_ids[start:end].unsqueeze(0).to(device)
        chunk_mask = torch.ones_like(chunk_ids)
        embeddings = get_embeddings(model, chunk_ids, chunk_mask, is_generative)
        mantissas = extract_mantissas(embeddings)
        if len(mantissas) > 0:
            chunk_stats.append(compute_mantissa_multi_stats(mantissas))

    if not chunk_stats:
        return {k: np.nan for k in ('ks', 'cvm', 'ad', 'benford_mse', 'benford_kl')}

    averaged = {}
    for key in ('ks', 'cvm', 'ad', 'benford_mse', 'benford_kl'):
        vals = [s[key] for s in chunk_stats if not np.isnan(s[key])]
        averaged[key] = float(np.mean(vals)) if vals else np.nan
    return averaged


def compute_mantissa_ks(text, model_name, is_generative=False):
    """Compute KS statistic for mantissa uniformity of model embeddings.

    Loads the model from disk on each call.  Prefer
    :func:`compute_ks_chunked` with pre-loaded models for batch work.

    Parameters
    ----------
    text : str
        Input text to embed.
    model_name : str
        HuggingFace model identifier.
    is_generative : bool
        If True, extract token embeddings directly (generative model).
        If False, use last hidden state (masked model).

    Returns
    -------
    float
        KS statistic vs uniform distribution, or np.nan on failure.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel, T5EncoderModel

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if 't5' in model_name.lower():
        model = T5EncoderModel.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)

    model = model.float().to(device)
    model.eval()

    return compute_ks_chunked(text, tokenizer, model, device, is_generative)


def fit_gaussian_4d(data, min_std=0.001, std_inflation=1.5):
    """Fit independent Gaussian per dimension with regularization.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    min_std : float
        Minimum standard deviation to avoid degenerate distributions.
    std_inflation : float
        Multiplicative factor applied to estimated std.  Values > 1 widen
        the decision boundary, improving cross-domain robustness at a small
        cost to within-domain accuracy.

    Returns
    -------
    dict
        {'mean': np.ndarray, 'std': np.ndarray}
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std = np.maximum(std, min_std) * std_inflation
    return {'mean': mean, 'std': std}


def evaluate(predictions, true_labels, posteriors=None):
    """Compute accuracy, precision, recall, F1, AUC-ROC, and confusion matrix.

    Parameters
    ----------
    predictions : np.ndarray of int
    true_labels : np.ndarray of int
    posteriors : np.ndarray of float, optional
        P(human | data) for each sample.  When provided, AUC-ROC is computed
        using P(AI) = 1 - posteriors as the score for class 1.

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1, auc_roc, confusion.
        auc_roc is np.nan when posteriors are not provided or when only
        one class is present.
    """
    tp = np.sum((predictions == 1) & (true_labels == 1))
    tn = np.sum((predictions == 0) & (true_labels == 0))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))

    accuracy = (tp + tn) / len(true_labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    auc_roc = np.nan
    if posteriors is not None and len(np.unique(true_labels)) == 2:
        scores = 1.0 - np.asarray(posteriors)
        auc_roc = _roc_auc(true_labels, scores)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'confusion': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
    }


def _roc_auc(y_true, scores):
    """Compute AUC-ROC via the trapezoidal rule.

    Parameters
    ----------
    y_true : array-like of int (0 or 1)
    scores : array-like of float
        Higher score => more likely class 1.

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)

    desc = np.argsort(-scores)
    y_sorted = y_true[desc]
    scores_sorted = scores[desc]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return np.nan

    tpr_prev, fpr_prev = 0.0, 0.0
    tp, fp = 0, 0
    auc = 0.0
    prev_score = scores_sorted[0] + 1

    for i in range(len(y_sorted)):
        if scores_sorted[i] != prev_score:
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
            tpr_prev, fpr_prev = tpr, fpr
            prev_score = scores_sorted[i]
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1

    tpr = tp / n_pos
    fpr = fp / n_neg
    auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
    return auc


def roc_curve(y_true, scores):
    """Compute ROC curve points for plotting.

    Parameters
    ----------
    y_true : array-like of int (0 or 1)
    scores : array-like of float
        Higher score => more likely class 1.

    Returns
    -------
    fpr : np.ndarray
    tpr : np.ndarray
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)

    desc = np.argsort(-scores)
    y_sorted = y_true[desc]
    scores_sorted = scores[desc]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    fprs, tprs = [0.0], [0.0]
    tp, fp = 0, 0
    prev_score = scores_sorted[0] + 1

    for i in range(len(y_sorted)):
        if scores_sorted[i] != prev_score:
            fprs.append(fp / n_neg)
            tprs.append(tp / n_pos)
            prev_score = scores_sorted[i]
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1

    fprs.append(fp / n_neg)
    tprs.append(tp / n_pos)
    return np.array(fprs), np.array(tprs)


def permutation_test(human_vals, ai_vals, n_permutations=10000):
    """Two-sided permutation test for difference in means.

    Parameters
    ----------
    human_vals : np.ndarray
    ai_vals : np.ndarray
    n_permutations : int

    Returns
    -------
    float
        p-value.
    """
    observed_diff = human_vals.mean() - ai_vals.mean()
    combined = np.concatenate([human_vals, ai_vals])
    n_human = len(human_vals)

    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = combined[:n_human].mean() - combined[n_human:].mean()
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    return count / n_permutations


def bootstrap_ci(data, n_bootstrap=10000, ci=95):
    """Bootstrap confidence interval for the mean.

    Parameters
    ----------
    data : np.ndarray
    n_bootstrap : int
    ci : float
        Confidence level in percent.

    Returns
    -------
    tuple
        (lower, upper) bounds.
    """
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(sample.mean())

    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return lower, upper
