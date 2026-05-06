def count_samples(ns_run, **kwargs):
    r"""Number of samples in run.

    Unlike most estimators this does not require log weights, but for
    convenience will not throw an error if they are specified.

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dict (see the data_processing module
        docstring for more details).

    Returns
    -------
    int
    """
    kwargs.pop('logw', None)
    kwargs.pop('simulate', None)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    return ns_run['logl'].shape[0]