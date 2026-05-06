def table_to_source_list(table, src_type=OutputSource):
    """
    Convert a table of data into a list of sources.

    A single table must have consistent source types given by src_type. src_type should be one of
    :class:`AegeanTools.models.OutputSource`, :class:`AegeanTools.models.SimpleSource`,
    or :class:`AegeanTools.models.IslandSource`.


    Parameters
    ----------
    table : Table
        Table of sources

    src_type : class
        Sources must be of type :class:`AegeanTools.models.OutputSource`,
        :class:`AegeanTools.models.SimpleSource`, or :class:`AegeanTools.models.IslandSource`.

    Returns
    -------
    sources : list
        A list of objects of the given type.
    """
    source_list = []
    if table is None:
        return source_list

    for row in table:
        # Initialise our object
        src = src_type()
        # look for the columns required by our source object
        for param in src_type.names:
            if param in table.colnames:
                # copy the value to our object
                val = row[param]
                # hack around float32's broken-ness
                if isinstance(val, np.float32):
                    val = np.float64(val)
                setattr(src, param, val)
        # save this object to our list of sources
        source_list.append(src)
    return source_list