def create_polynoms():
    """Create and return poly1d objects.

    Uses the parameters from Morgan to create poly1d objects for
    calculations.
    """
    fname = pr.resource_filename('pyciss', 'data/soliton_prediction_parameters.csv')
    res_df = pd.read_csv(fname)
    polys = {}
    for resorder, row in zip('65 54 43 21'.split(),
                             range(4)):
        p = poly1d([res_df.loc[row, 'Slope (km/yr)'], res_df.loc[row, 'Intercept (km)']])
        polys['janus ' + ':'.join(resorder)] = p
    return polys