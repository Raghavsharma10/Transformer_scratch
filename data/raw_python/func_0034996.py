def create_transition(oracle, method='trn'):
    """Create a transition matrix based on oracle links"""
    mat, hist, n = _create_trn_mat_symbolic(oracle, method)
    return mat, hist, n