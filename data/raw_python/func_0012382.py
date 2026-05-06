def average_by_key(dict_in, key):
    """Helper function for plot_run_nlive.

    Try returning the average of dict_in[key] and, if this does not work or if
    key is None, return average of whole dict.

    Parameters
    ----------
    dict_in: dict
        Values should be arrays.
    key: str

    Returns
    -------
    average: float
    """
    if key is None:
        return np.mean(np.concatenate(list(dict_in.values())))
    else:
        try:
            return np.mean(dict_in[key])
        except KeyError:
            print('method name "' + key + '" not found, so ' +
                  'normalise area under the analytic relative posterior ' +
                  'mass curve using the mean of all methods.')
            return np.mean(np.concatenate(list(dict_in.values())))