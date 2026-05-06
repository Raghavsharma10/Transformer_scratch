def nr_of_baselines(na, auto_correlations=False):
    """
    Compute the number of baselines for the
    given number of antenna. Can specify whether
    auto-correlations should be taken into
    account
    """
    m = (na-1) if auto_correlations is False else (na+1)
    return (na*m)//2