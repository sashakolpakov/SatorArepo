import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
from .utils import (
    extract_leading_digits,
    calculate_empirical_distribution,
    compute_benford_indicators,
)

# MPS setup: enable fallback for unsupported ops
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

GENERATIVE_MODELS = [
    'EleutherAI/gpt-neo-125m',                    # 768 dims - GPT-Neo
    'flax-community/gpt-neo-125M-code-clippy',    # 768 dims - GPT-Neo code fine-tune
]

MASKED_MODELS = [
    'distilbert-base-uncased',                     # 768 dims - BERT
    'google/electra-base-discriminator',              # 768 dims - ELECTRA
]

DEFAULT_MODELS = GENERATIVE_MODELS + MASKED_MODELS

def generate_embeddings(text, model_list=None):
    """Generate embeddings from multiple models and flatten them properly."""
    if model_list is None:
        model_list = DEFAULT_MODELS

    embeddings = {}
    for model_name in model_list:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Use T5EncoderModel for T5 to avoid decoder deprecation warnings
            if 't5' in model_name.lower():
                model = T5EncoderModel.from_pretrained(model_name)
            else:
                model = AutoModel.from_pretrained(model_name)

            # Use float32 for stability (avoid NaN issues with float16 on MPS)
            model = model.float().to(device)

            # Handle models without pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # Get embeddings - handle different model output formats
            if hasattr(outputs, 'last_hidden_state'):
                emb = outputs.last_hidden_state[0].cpu()  # Shape: (num_tokens, embedding_dim)
            elif hasattr(outputs, 'encoder_last_hidden_state'):
                emb = outputs.encoder_last_hidden_state[0].cpu()  # T5 encoder output
            else:
                emb = outputs[0][0].cpu()

            # CRITICAL: Flatten the entire embedding matrix into a single vector
            # Following BENADV (Wang et al. 2025): U[j]_i = flatten(E[j]_i)
            flat_emb = emb.flatten()

            # Check for NaN values
            if torch.isnan(flat_emb).any():
                print(f"  {model_name}: WARNING - NaN values detected, skipping model")
                embeddings[model_name] = None
            else:
                embeddings[model_name] = flat_emb
                print(f"  {model_name}: extracted {flat_emb.shape[0]} embedding values, range [{flat_emb.min():.4f}, {flat_emb.max():.4f}]")

            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

        except Exception as e:
            print(f"  Warning: Failed to load {model_name}: {e}")
            embeddings[model_name] = None

    return embeddings

def construct_feature_vector(text, model_list=None):
    """
    Construct feature vector following BENADV methodology.
    Returns 4 features per model: [MSE, R², Chi², KL]
    """
    if model_list is None:
        model_list = DEFAULT_MODELS

    all_features = []
    embeddings_dict = generate_embeddings(text, model_list)

    for model_name in model_list:
        emb = embeddings_dict.get(model_name)

        # If model failed, add zeros
        if emb is None or len(emb) < 1:
            all_features.extend([0] * 4)
            continue

        # Extract leading digits from flattened embedding vector
        digits = extract_leading_digits(emb)

        if len(digits) == 0:
            all_features.extend([0] * 4)
            continue

        # Calculate empirical distribution
        p_hat = calculate_empirical_distribution(digits)

        # Compute Benford indicators (4 features: MSE, R², Chi², KL)
        ind = compute_benford_indicators(p_hat)

        # Use only the 4 main indicators like BENADV paper
        all_features.extend([ind['MSE'], ind['R2'], ind['Chi2'], ind['KL']])

    return np.array(all_features)


def extract_4d_features_with_indices(texts, model_list=None):
    """Extract 4D mantissa-KS feature vectors and surviving row indices.

    Rows with failed/NaN features are dropped, and their original positions
    are omitted from the returned index array.
    """
    from .stats import compute_ks_chunked

    if model_list is None:
        model_list = DEFAULT_MODELS

    # Pre-load models once instead of per text×model
    loaded = {}
    for model_name in model_list:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if 't5' in model_name.lower():
            m = T5EncoderModel.from_pretrained(model_name)
        else:
            m = AutoModel.from_pretrained(model_name)
        m = m.float().to(device)
        m.eval()
        loaded[model_name] = (tokenizer, m)

    features = []
    valid_indices = []
    for i, text in enumerate(texts):
        if i % 10 == 0:
            print(f"  {i+1}/{len(texts)}...", flush=True)
        try:
            vec = []
            for model_name in model_list:
                is_gen = model_name in GENERATIVE_MODELS
                tok, mdl = loaded[model_name]
                ks = compute_ks_chunked(text, tok, mdl, device, is_gen)
                vec.append(ks)

            if len(vec) == len(model_list) and not any(np.isnan(vec)):
                features.append(vec)
                valid_indices.append(i)
        except Exception as e:
            continue

    # Clean up
    del loaded
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    feature_array = np.array(features) if features else np.empty((0, len(model_list)))
    return feature_array, np.array(valid_indices, dtype=int)


def extract_4d_features(texts, model_list=None):
    """Extract 4D mantissa-KS feature vectors (production pipeline).

    This is the feature extraction used by the production MoE engine.
    Each text gets one KS statistic per model, yielding a 4D vector.

    For the auxiliary Benford-indicator approach (MSE, R2, Chi2, KL per model),
    see ``construct_feature_vector()``.

    Parameters
    ----------
    texts : list of str
        Input texts.
    model_list : list of str or None
        Model names. Defaults to DEFAULT_MODELS.

    Returns
    -------
    np.ndarray, shape (n_valid, len(model_list))
        Feature matrix (rows with any NaN are dropped).
    """
    features, _ = extract_4d_features_with_indices(texts, model_list=model_list)
    return features


STAT_NAMES = ('ks', 'cvm', 'ad', 'benford_mse', 'benford_kl')


def extract_multi_stat_features(texts, model_list=None, stat_keys=None):
    """Extract multi-statistic features per model.

    Computes up to 5 statistics per model from mantissa distributions,
    yielding a feature vector of len(stat_keys) * len(model_list).

    Parameters
    ----------
    texts : list of str
        Input texts.
    model_list : list of str or None
        Model names. Defaults to DEFAULT_MODELS.
    stat_keys : tuple of str or None
        Which statistics to include. Defaults to all 5:
        ('ks', 'cvm', 'ad', 'benford_mse', 'benford_kl').

    Returns
    -------
    np.ndarray, shape (n_valid, len(stat_keys) * len(model_list))
        Feature matrix (rows with any NaN are dropped).
    """
    from .stats import compute_multi_stats_chunked

    if model_list is None:
        model_list = DEFAULT_MODELS
    if stat_keys is None:
        stat_keys = STAT_NAMES

    # Pre-load models once instead of per text×model
    loaded = {}
    for model_name in model_list:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if 't5' in model_name.lower():
            m = T5EncoderModel.from_pretrained(model_name)
        else:
            m = AutoModel.from_pretrained(model_name)
        m = m.float().to(device)
        m.eval()
        loaded[model_name] = (tokenizer, m)

    features = []
    for i, text in enumerate(texts):
        if i % 10 == 0:
            print(f"  {i+1}/{len(texts)}...", flush=True)
        try:
            vec = []
            for model_name in model_list:
                is_gen = model_name in GENERATIVE_MODELS
                tok, mdl = loaded[model_name]
                multi = compute_multi_stats_chunked(
                    text, tok, mdl, device, is_gen,
                )
                for key in stat_keys:
                    vec.append(multi[key])

            if len(vec) == len(stat_keys) * len(model_list) and not any(np.isnan(v) for v in vec):
                features.append(vec)
        except Exception:
            continue

    # Clean up
    del loaded
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    n_features = len(stat_keys) * len(model_list)
    return np.array(features) if features else np.empty((0, n_features))
