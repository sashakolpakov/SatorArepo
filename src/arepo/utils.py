import numpy as np
from scipy import stats

def extract_leading_digits(embedding_tensor):
    """
    Extract the first non-zero digit from each number.
    No scaling - just scan left to right for first non-zero digit!
    """
    # Convert to numpy if it's a torch tensor
    if hasattr(embedding_tensor, 'numpy'):
        flat_values = embedding_tensor.flatten().numpy()
    else:
        flat_values = np.array(embedding_tensor).flatten()

    # Filter out NaN and infinite values
    flat_values = flat_values[np.isfinite(flat_values)]

    # Take absolute values and filter out zeros
    abs_values = np.abs(flat_values)
    abs_values = abs_values[abs_values > 0]

    if len(abs_values) == 0:
        print(f"  Warning: All values are zero")
        return np.array([])

    leading_digits = []
    for val in abs_values:
        # Convert to string and scan for first non-zero digit
        val_str = f"{val:.20e}"  # Scientific notation to handle any magnitude

        # Extract mantissa and find first non-zero digit
        # Format: "1.234e-05" or "5.678e+02"
        mantissa = val_str.split('e')[0].replace('.', '').replace('-', '')

        # Find first non-zero digit
        for char in mantissa:
            if char.isdigit() and char != '0':
                leading_digits.append(int(char))
                break

    if len(leading_digits) == 0:
        print(f"  Warning: No valid leading digits extracted from {len(abs_values)} values")

    return np.array(leading_digits)

def calculate_empirical_distribution(digit_array):
    counts = np.bincount(digit_array, minlength=10)[1:]
    total = counts.sum()
    return counts / total if total > 0 else np.zeros(9)

def compute_benford_indicators(p):
    digits = np.arange(1, 10)
    q = np.log10(1 + 1/digits)

    epsilon = 1e-10
    p, q = np.clip(p, epsilon, 1), np.clip(q, epsilon, 1)

    mse = np.mean((p - q) ** 2)
    ss_res = np.sum((p - q) ** 2)
    ss_tot = np.sum((p - np.mean(p)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    n = 1000
    chi2 = np.sum(((p*n - q*n) ** 2) / np.clip(q*n, 1e-10, np.inf))
    kl = np.sum(p * np.log(p / q))

    return {'MSE': mse, 'R2': r2, 'Chi2': chi2, 'KL': kl}
