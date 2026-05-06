def get_tracerinfo(tracerinfo_file):
    """
    Read an output's tracerinfo.dat file and parse into a DataFrame for
    use in selecting and parsing categories.

    Parameters
    ----------
    tracerinfo_file : str
        Path to tracerinfo.dat

    Returns
    -------
    DataFrame containing the tracer information.

    """

    widths = [rec.width for rec in tracer_recs]
    col_names = [rec.name for rec in tracer_recs]
    dtypes = [rec.type for rec in tracer_recs]
    usecols = [name for name in col_names if not name.startswith('-')]

    tracer_df = pd.read_fwf(tracerinfo_file, widths=widths, names=col_names,
                            dtypes=dtypes, comment="#", header=None,
                            usecols=usecols)

    # Check an edge case related to a bug in GEOS-Chem v12.0.3 which 
    # erroneously dropped short/long tracer names in certain tracerinfo.dat outputs.
    # What we do here is figure out which rows were erroneously processed (they'll 
    # have NaNs in them) and raise a warning if there are any
    na_free = tracer_df.dropna(subset=['tracer', 'scale'])
    only_na = tracer_df[~tracer_df.index.isin(na_free.index)]
    if len(only_na) > 0:
        warn("At least one row in {} wasn't decoded correctly; we strongly"
             " recommend you manually check that file to see that all"
             " tracers are properly recorded."
             .format(tracerinfo_file)) 

    tracer_desc = {tracer.name: tracer.desc for tracer in tracer_recs
                   if not tracer.name.startswith('-')}

    # Process some of the information about which variables are hydrocarbons
    # and chemical tracers versus other diagnostics.
    def _assign_hydrocarbon(row):
        if row['C'] != 1:
            row['hydrocarbon'] = True
            row['molwt'] = C_MOLECULAR_WEIGHT
        else:
            row['hydrocarbon'] = False
        return row

    tracer_df = (
        tracer_df
            .apply(_assign_hydrocarbon, axis=1)
            .assign(chemical=lambda x: x['molwt'].astype(bool))
    )

    return tracer_df, tracer_desc