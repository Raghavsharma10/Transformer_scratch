def float_str(f, min_digits=2, max_digits=6):
    """
    Returns a string representing a float, where the number of
    significant digits is min_digits unless it takes more digits
    to hit a non-zero digit (and the number is 0 < x < 1).
    We stop looking for a non-zero digit after max_digits.
    """
    if f >= 1 or f <= 0:
        return str(round_float(f, min_digits))
    start_str = str(round_float(f, max_digits))
    digits = start_str.split(".")[1]
    non_zero_indices = []
    for i, digit in enumerate(digits):
        if digit != "0":
            non_zero_indices.append(i + 1)
    # Only saw 0s.
    if len(non_zero_indices) == 0:
        num_digits = min_digits
    else:
        # Of the non-zero digits, pick the num_digit'th of those (including any zeros)
        min_non_zero_indices = range(non_zero_indices[0], non_zero_indices[-1] + 1)[:min_digits]
        num_digits = min_non_zero_indices[-1]
    return str(round_float(f, num_digits))