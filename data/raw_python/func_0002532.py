def jsd_df_to_2d(jsd_df):
    """Transform a tall JSD dataframe to a square matrix of mean JSDs

    Parameters
    ----------
    jsd_df : pandas.DataFrame
        A (n_features, n_phenotypes^2) dataframe of the JSD between each
        feature between and within phenotypes

    Returns
    -------
    jsd_2d : pandas.DataFrame
        A (n_phenotypes, n_phenotypes) symmetric dataframe of the mean JSD
        between and within phenotypes
    """
    jsd_2d = jsd_df.mean().reset_index()
    jsd_2d = jsd_2d.rename(
        columns={'level_0': 'phenotype1', 'level_1': 'phenotype2', 0: 'jsd'})
    jsd_2d = jsd_2d.pivot(index='phenotype1', columns='phenotype2',
                          values='jsd')
    return jsd_2d + np.tril(jsd_2d.T, -1)