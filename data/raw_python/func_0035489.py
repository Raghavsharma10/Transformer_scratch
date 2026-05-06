def _fill_diagonals(m, diag_indices):
    """Fills diagonals of `nsites` matrices in `m` so rows sum to 0."""
    assert m.ndim == 3, "M must have 3 dimensions"
    assert m.shape[1] == m.shape[2], "M must contain square matrices"
    for r in range(m.shape[0]):
        scipy.fill_diagonal(m[r], 0)
        m[r][diag_indices] -= scipy.sum(m[r], axis=1)