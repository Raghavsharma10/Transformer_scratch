def _partition(iter_dims, data_sources):
    """
    Partition data sources into

    1. Dictionary of data sources associated with radio sources.
    2. List of data sources to feed multiple times.
    3. List of data sources to feed once.
    """

    src_nr_vars = set(source_var_types().values())
    iter_dims = set(iter_dims)

    src_data_sources = collections.defaultdict(list)
    feed_many = []
    feed_once = []

    for ds in data_sources:
        # Is this data source associated with
        # a radio source (point, gaussian, etc.?)
        src_int = src_nr_vars.intersection(ds.shape)

        if len(src_int) > 1:
            raise ValueError("Data source '{}' contains multiple "
                            "source types '{}'".format(ds.name, src_int))
        elif len(src_int) == 1:
            # Yep, record appropriately and iterate
            src_data_sources[src_int.pop()].append(ds)
            continue

        # Are we feeding this data source multiple times
        # (Does it possess dimensions on which we iterate?)
        if len(iter_dims.intersection(ds.shape)) > 0:
            feed_many.append(ds)
            continue

        # Assume this is a data source that we only feed once
        feed_once.append(ds)

    return src_data_sources, feed_many, feed_once