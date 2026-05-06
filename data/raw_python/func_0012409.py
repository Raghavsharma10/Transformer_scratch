def combine_ns_runs(run_list_in, **kwargs):
    """
    Combine a list of complete nested sampling run dictionaries into a single
    ns run.

    Input runs must contain any repeated threads.

    Parameters
    ----------
    run_list_in: list of dicts
        List of nested sampling runs in dict format (see data_processing module
        docstring for more details).
    kwargs: dict, optional
        Options for check_ns_run.

    Returns
    -------
    run: dict
        Nested sampling run dict (see data_processing module docstring for more
        details).
    """
    run_list = copy.deepcopy(run_list_in)
    if len(run_list) == 1:
        run = run_list[0]
    else:
        nthread_tot = 0
        for i, _ in enumerate(run_list):
            check_ns_run(run_list[i], **kwargs)
            run_list[i]['thread_labels'] += nthread_tot
            nthread_tot += run_list[i]['thread_min_max'].shape[0]
        thread_min_max = np.vstack([run['thread_min_max'] for run in run_list])
        # construct samples array from the threads, including an updated nlive
        samples_temp = np.vstack([array_given_run(run) for run in run_list])
        samples_temp = samples_temp[np.argsort(samples_temp[:, 0])]
        # Make combined run
        run = dict_given_run_array(samples_temp, thread_min_max)
        # Combine only the additive properties stored in run['output']
        run['output'] = {}
        for key in ['nlike', 'ndead']:
            try:
                run['output'][key] = sum([temp['output'][key] for temp in
                                          run_list_in])
            except KeyError:
                pass
    check_ns_run(run, **kwargs)
    return run