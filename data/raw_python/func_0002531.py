def cross_phenotype_jsd(data, groupby, bins, n_iter=100):
    """Jensen-Shannon divergence of features across phenotypes

    Parameters
    ----------
    data : pandas.DataFrame
        A (n_samples, n_features) Dataframe
    groupby : mappable
        A samples to phenotypes mapping
    n_iter : int
        Number of bootstrap resampling iterations to perform for the
        within-group comparisons
    n_bins : int
        Number of bins to binify the singles data on

    Returns
    -------
    jsd_df : pandas.DataFrame
        A (n_features, n_phenotypes^2) dataframe of the JSD between each
        feature between and within phenotypes
    """
    grouped = data.groupby(groupby)
    jsds = []

    seen = set([])

    for phenotype1, df1 in grouped:
        for phenotype2, df2 in grouped:
            pair = tuple(sorted([phenotype1, phenotype2]))
            if pair in seen:
                continue
            seen.add(pair)

            if phenotype1 == phenotype2:
                seriess = []
                bs = cross_validation.Bootstrap(df1.shape[0], n_iter=n_iter,
                                                train_size=0.5)
                for i, (ind1, ind2) in enumerate(bs):
                    df1_subset = df1.iloc[ind1, :]
                    df2_subset = df2.iloc[ind2, :]
                    seriess.append(
                        binify_and_jsd(df1_subset, df2_subset, None, bins))
                series = pd.concat(seriess, axis=1, names=None).mean(axis=1)
                series.name = pair
                jsds.append(series)
            else:
                series = binify_and_jsd(df1, df2, pair, bins)
                jsds.append(series)
    return pd.concat(jsds, axis=1)