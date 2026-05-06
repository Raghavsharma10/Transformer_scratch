def pairwise_dists_on_cols(df_in, earth_mover_dist=True, energy_dist=True):
    """Computes pairwise statistical distance measures.

    parameters
    ----------
    df_in: pandas data frame
        Columns represent estimators and rows represent runs.
        Each data frane element is an array of values which are used as samples
        in the distance measures.
    earth_mover_dist: bool, optional
        Passed to error_analysis.pairwise_distances.
    energy_dist: bool, optional
        Passed to error_analysis.pairwise_distances.

    returns
    -------
    df: pandas data frame with kl values for each pair.
    """
    df = pd.DataFrame()
    for col in df_in.columns:
        df[col] = nestcheck.error_analysis.pairwise_distances(
            df_in[col].values, earth_mover_dist=earth_mover_dist,
            energy_dist=energy_dist)
    return df