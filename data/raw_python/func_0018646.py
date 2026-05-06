def read_scen_file(
    filepath,
    columns={
        "model": ["unspecified"],
        "scenario": ["unspecified"],
        "climate_model": ["unspecified"],
    },
    **kwargs
):
    """
    Read a MAGICC .SCEN file.

    Parameters
    ----------
    filepath : str
        Filepath of the .SCEN file to read

    columns : dict
        Passed to ``__init__`` method of MAGICCData. See
        ``MAGICCData.__init__`` for details.

    kwargs
        Passed to ``__init__`` method of MAGICCData. See
        ``MAGICCData.__init__`` for details.

    Returns
    -------
    :obj:`pymagicc.io.MAGICCData`
        ``MAGICCData`` object containing the data and metadata.
    """
    mdata = MAGICCData(filepath, columns=columns, **kwargs)

    return mdata