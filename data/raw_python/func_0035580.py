def modelComparisonDataFrame(modelcomparisonfile, splitparams):
    """Converts ``modelcomparison.md`` file to `pandas` DataFrame.

    Running ``phydms_comprehensive`` creates a file with the suffix
    ``modelcomparison.md``. This function converts that file into a
    DataFrame that is easy to handle for downstream analysis.

    Args:
        `modelcomparisonfile` (str)
            The name of the ``modelcomparison.md`` file.
        `splitparams` (bool)
            If `True`, create a new column for each model param in
            the `ParamValues` column, with values of `NaN` if that
            model does not have such a parameter.

    Returns:
        A `pandas` DataFrame with the information in the model
        comparison file.

    >>> with tempfile.NamedTemporaryFile(mode='w') as f:
    ...     _ = f.write('\\n'.join([
    ...         '| Model | deltaAIC | LogLikelihood | nParams | ParamValues  |',
    ...         '|-------|----------|---------------|---------|--------------|',
    ...         '| ExpCM | 0.00     | -1000.00      | 7       | x=1.0, y=2.0 |',
    ...         '| YNGKP | 10.2     | -1005.10      | 7       | x=1.3, z=0.1 |',
    ...         ]))
    ...     f.flush()
    ...     df_split = modelComparisonDataFrame(f.name, splitparams=True)
    ...     df_nosplit = modelComparisonDataFrame(f.name, splitparams=False)
    >>> df_nosplit.equals(pandas.DataFrame.from_records(
    ...         [['ExpCM', 0, -1000, 7, 'x=1.0, y=2.0'],
    ...          ['YNGKP', 10.2, -1005.1, 7, 'x=1.3, z=0.1']],
    ...         columns=['Model', 'deltaAIC', 'LogLikelihood',
    ...                  'nParams', 'ParamValues']))
    True
    >>> df_split.equals(pandas.DataFrame.from_records(
    ...         [['ExpCM', 0, -1000, 7, 1.0, 2.0, numpy.nan],
    ...          ['YNGKP', 10.2, -1005.1, 7, 1.3, numpy.nan, 0.1]],
    ...         columns=['Model', 'deltaAIC', 'LogLikelihood',
    ...                  'nParams', 'x', 'y', 'z']))
    True
    """
    df = (pandas.read_csv(modelcomparisonfile, sep='|', skiprows=[1])
            .select(lambda x: 'Unnamed' not in x, axis=1)
            )

    # strip whitespace
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if pandas.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()

    paramsdict = {}
    if splitparams:
        for (i, paramstr) in df['ParamValues'].iteritems():
            paramsdict[i] = dict(map(lambda tup: (tup[0], float(tup[1])),
                    [param.strip().split('=') for param in paramstr.split(',')]))
        params_df = pandas.DataFrame.from_dict(paramsdict, orient='index')
        params_df = params_df[sorted(params_df.columns)]
        df = (df.join(params_df)
                .drop('ParamValues', axis=1)
                )

    return df