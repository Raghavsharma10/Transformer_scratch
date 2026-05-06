def _dist_obs_oracle(oracle, query, trn_list):
    """A helper function calculating distances between a feature and frames in oracle."""
    a = np.subtract(query, [oracle.f_array[t] for t in trn_list])
    return (a * a).sum(axis=1)