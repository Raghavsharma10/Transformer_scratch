def array_given_run(ns_run):
    """Converts information on samples in a nested sampling run dictionary into
    a numpy array representation. This allows fast addition of more samples and
    recalculation of nlive.

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dict (see data_processing module docstring for more
        details).

    Returns
    -------
    samples: 2d numpy array
        Array containing columns
        [logl, thread label, change in nlive at sample, (thetas)]
        with each row representing a single sample.
    """
    samples = np.zeros((ns_run['logl'].shape[0], 3 + ns_run['theta'].shape[1]))
    samples[:, 0] = ns_run['logl']
    samples[:, 1] = ns_run['thread_labels']
    # Calculate 'change in nlive' after each step
    samples[:-1, 2] = np.diff(ns_run['nlive_array'])
    samples[-1, 2] = -1  # nlive drops to zero after final point
    samples[:, 3:] = ns_run['theta']
    return samples