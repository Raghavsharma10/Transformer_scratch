def write_run_output(run, **kwargs):
    """Writes PolyChord output files corresponding to the input nested sampling
    run. The file root is

    .. code-block:: python

        root = os.path.join(run['output']['base_dir'],
                            run['output']['file_root'])

    Output files which can be made with this function (see the PolyChord
    documentation for more information about what each contains):

    * [root].stats
    * [root].txt
    * [root]_equal_weights.txt
    * [root]_dead-birth.txt
    * [root]_dead.txt

    Files produced by PolyChord which are not made by this function:

    * [root].resume: for resuming runs part way through (not relevant for a
      completed run).
    * [root]_phys_live.txt and [root]phys_live-birth.txt: for checking runtime
      progress (not relevant for a completed run).
    * [root].paramnames: for use with getdist (not needed when calling getdist
      from within python).

    Parameters
    ----------
    ns_run: dict
        Nested sampling run dict (see data_processing module docstring for more
        details).
    write_dead: bool, optional
        Whether or not to write [root]_dead.txt and [root]_dead-birth.txt.
    write_stats: bool, optional
        Whether or not to write [root].stats.
    posteriors: bool, optional
        Whether or not to write [root].txt.
    equals: bool, optional
        Whether or not to write [root]_equal_weights.txt.
    stats_means_errs: bool, optional
        Whether or not to calculate mean values of :math:`\log \mathcal{Z}` and
        each parameter, and their uncertainties.
    fmt: str, optional
        Formatting for numbers written by np.savetxt. Default value is set to
        make output files look like the ones produced by PolyChord.
    n_simulate: int, optional
        Number of bootstrap replications to use when estimating uncertainty on
        evidence and parameter means.
    """
    write_dead = kwargs.pop('write_dead', True)
    write_stats = kwargs.pop('write_stats', True)
    posteriors = kwargs.pop('posteriors', False)
    equals = kwargs.pop('equals', False)
    stats_means_errs = kwargs.pop('stats_means_errs', True)
    fmt = kwargs.pop('fmt', '% .14E')
    n_simulate = kwargs.pop('n_simulate', 100)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    mandatory_keys = ['file_root', 'base_dir']
    for key in mandatory_keys:
        assert key in run['output'], key + ' not in run["output"]'
    root = os.path.join(run['output']['base_dir'], run['output']['file_root'])
    if write_dead:
        samples = run_dead_birth_array(run)
        np.savetxt(root + '_dead-birth.txt', samples, fmt=fmt)
        np.savetxt(root + '_dead.txt', samples[:, :-1], fmt=fmt)
    if equals or posteriors:
        w_rel = nestcheck.ns_run_utils.get_w_rel(run)
        post_arr = np.zeros((run['theta'].shape[0], run['theta'].shape[1] + 2))
        post_arr[:, 0] = w_rel
        post_arr[:, 1] = -2 * run['logl']
        post_arr[:, 2:] = run['theta']
    if posteriors:
        np.savetxt(root + '.txt', post_arr, fmt=fmt)
        run['output']['nposterior'] = post_arr.shape[0]
    else:
        run['output']['nposterior'] = 0
    if equals:
        inds = np.where(w_rel > np.random.random(w_rel.shape[0]))[0]
        np.savetxt(root + '_equal_weights.txt', post_arr[inds, 1:],
                   fmt=fmt)
        run['output']['nequals'] = inds.shape[0]
    else:
        run['output']['nequals'] = 0
    if write_stats:
        run['output']['ndead'] = run['logl'].shape[0]
        if stats_means_errs:
            # Get logZ and param estimates and errors
            estimators = [e.logz]
            for i in range(run['theta'].shape[1]):
                estimators.append(functools.partial(e.param_mean, param_ind=i))
            values = nestcheck.ns_run_utils.run_estimators(run, estimators)
            stds = nestcheck.error_analysis.run_std_bootstrap(
                run, estimators, n_simulate=n_simulate)
            run['output']['logZ'] = values[0]
            run['output']['logZerr'] = stds[0]
            run['output']['param_means'] = list(values[1:])
            run['output']['param_mean_errs'] = list(stds[1:])
        write_stats_file(run['output'])