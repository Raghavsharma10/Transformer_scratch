def readDivPressure(fileName):
    """Reads in diversifying pressures from some file.

    Scale diversifying pressure values so absolute value of the max value is 1,
    unless all values are zero.

    Args:
        `fileName` (string or readable file-like object)
            File holding diversifying pressure values. Can be
            comma-, space-, or tab-separated file. The first column
            is the site (consecutively numbered, sites starting
            with one) and the second column is the diversifying pressure values.

    Returns:
        `divPressure` (dict keyed by ints)
            `divPressure[r][v]` is the diversifying pressure value of site `r`.
    """
    try:
        df = pandas.read_csv(fileName, sep=None, engine='python')
        pandasformat = True
    except ValueError:
        pandasformat = False
    df.columns = ['site', 'divPressureValue']
    scaleFactor = max(df["divPressureValue"].abs())
    if scaleFactor > 0:
        df["divPressureValue"] = [x / scaleFactor for x in df["divPressureValue"]]
    assert len(df['site'].tolist()) == len(set(df['site'].tolist())),"There is at least one non-unique site in {0}".format(fileName)
    assert max(df["divPressureValue"].abs()) <= 1, "The scaling produced a diversifying pressure value with an absolute value greater than one."
    sites = df['site'].tolist()
    divPressure = {}
    for r in sites:
        divPressure[r] = df[df['site'] == r]["divPressureValue"].tolist()[0]
    return divPressure