def rankagg(df, method="stuart"):
    """Return aggregated ranks.

    Implementation is ported from the RobustRankAggreg R package
    
    References: 
        Kolde et al., 2012, DOI: 10.1093/bioinformatics/btr709
        Stuart et al., 2003,  DOI: 10.1126/science.1087447

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with values to be ranked and aggregated

    Returns
    -------
    pandas.DataFrame with aggregated ranks
    """
    rmat = pd.DataFrame(index=df.iloc[:,0])

    step = 1 / rmat.shape[0]
    for col in df.columns:
        rmat[col] = pd.DataFrame({col:np.arange(step, 1 + step, step)}, index=df[col]).loc[rmat.index]
    rmat = rmat.apply(sorted, 1, result_type="expand")
    p = rmat.apply(qStuart, 1)
    df = pd.DataFrame(
        {"p.adjust":multipletests(p, method="h")[1]}, 
        index=rmat.index).sort_values('p.adjust')     
    
    return df["p.adjust"]