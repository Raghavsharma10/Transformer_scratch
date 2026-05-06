def writeDB(filename, catalog, meta=None):
    """
    Output an sqlite3 database containing one table for each source type

    Parameters
    ----------
    filename : str
        Output filename

    catalog : list
        List of sources of type :class:`AegeanTools.models.OutputSource`,
        :class:`AegeanTools.models.SimpleSource`, or :class:`AegeanTools.models.IslandSource`.

    meta : dict
        Meta data to be written to table `meta`

    Returns
    -------
    None
    """

    def sqlTypes(obj, names):
        """
        Return the sql type corresponding to each named parameter in obj
        """
        types = []
        for n in names:
            val = getattr(obj, n)
            if isinstance(val, bool):
                types.append("BOOL")
            elif isinstance(val, (int, np.int64, np.int32)):
                types.append("INT")
            elif isinstance(val, (float, np.float64, np.float32)):  # float32 is bugged and claims not to be a float
                types.append("FLOAT")
            elif isinstance(val, six.string_types):
                types.append("VARCHAR")
            else:
                log.warning("Column {0} is of unknown type {1}".format(n, type(n)))
                log.warning("Using VARCHAR")
                types.append("VARCHAR")
        return types

    if os.path.exists(filename):
        log.warning("overwriting {0}".format(filename))
        os.remove(filename)
    conn = sqlite3.connect(filename)
    db = conn.cursor()
    # determine the column names by inspecting the catalog class
    for t, tn in zip(classify_catalog(catalog), ["components", "islands", "simples"]):
        if len(t) < 1:
            continue  #don't write empty tables
        col_names = t[0].names
        col_types = sqlTypes(t[0], col_names)
        stmnt = ','.join(["{0} {1}".format(a, b) for a, b in zip(col_names, col_types)])
        db.execute('CREATE TABLE {0} ({1})'.format(tn, stmnt))
        stmnt = 'INSERT INTO {0} ({1}) VALUES ({2})'.format(tn, ','.join(col_names), ','.join(['?' for i in col_names]))
        # expend the iterators that are created by python 3+
        data = list(map(nulls, list(r.as_list() for r in t)))
        db.executemany(stmnt, data)
        log.info("Created table {0}".format(tn))
    # metadata add some meta data
    db.execute("CREATE TABLE meta (key VARCHAR, val VARCHAR)")
    for k in meta:
        db.execute("INSERT INTO meta (key, val) VALUES (?,?)", (k, meta[k]))
    conn.commit()
    log.info(db.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall())
    conn.close()
    log.info("Wrote file {0}".format(filename))
    return