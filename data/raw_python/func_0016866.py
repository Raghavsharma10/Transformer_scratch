def default_base_ant_pairs(self, context):
    """ Compute base antenna pairs """
    k = 0 if context.cfg['auto_correlations'] == True else 1
    na = context.dim_global_size('na')
    gen = (i.astype(context.dtype) for i in np.triu_indices(na, k))

    # Cache np.triu_indices(na, k) as its likely that (na, k) will
    # stay constant much of the time. Assumption here is that this
    # method will be grafted onto a DefaultsSourceProvider with
    # the appropriate members.
    if self._is_cached:
        array_cache = self._chunk_cache['default_base_ant_pairs']
        key = (k, na)

        # Cache miss
        if key not in array_cache:
            array_cache[key] = tuple(gen)

        return array_cache[key]

    return tuple(gen)