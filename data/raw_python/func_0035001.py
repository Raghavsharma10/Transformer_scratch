def _query_init(k, oracle, query, method='all'):
    """A helper function for query-matching function initialization."""
    if method == 'all':
        a = np.subtract(query, [oracle.f_array[t] for t in oracle.latent[oracle.data[k]]])
        dvec = (a * a).sum(axis=1)  # Could skip the sqrt
        _d = dvec.argmin()
        return oracle.latent[oracle.data[k]][_d], dvec[_d]

    else:
        a = np.subtract(query, oracle.f_array[k])
        dvec = (a * a).sum()  # Could skip the sqrt
        return k, dvec