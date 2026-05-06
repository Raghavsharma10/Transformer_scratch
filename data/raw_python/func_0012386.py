def process_multinest_run(file_root, base_dir, **kwargs):
    """Loads data from a MultiNest run into the nestcheck dictionary format for
    analysis.

    N.B. producing required output file containing information about the
    iso-likelihood contours within which points were sampled (where they were
    "born") requies MultiNest version 3.11 or later.

    Parameters
    ----------
    file_root: str
        Root name for output files. When running MultiNest, this is determined
        by the nest_root parameter.
    base_dir: str
        Directory containing output files. When running MultiNest, this is
        determined by the nest_root parameter.
    kwargs: dict, optional
        Passed to ns_run_utils.check_ns_run (via process_samples_array)


    Returns
    -------
    ns_run: dict
        Nested sampling run dict (see the module docstring for more details).
    """
    # Load dead and live points
    dead = np.loadtxt(os.path.join(base_dir, file_root) + '-dead-birth.txt')
    live = np.loadtxt(os.path.join(base_dir, file_root)
                      + '-phys_live-birth.txt')
    # Remove unnecessary final columns
    dead = dead[:, :-2]
    live = live[:, :-1]
    assert dead[:, -2].max() < live[:, -2].min(), (
        'final live points should have greater logls than any dead point!',
        dead, live)
    ns_run = process_samples_array(np.vstack((dead, live)), **kwargs)
    assert np.all(ns_run['thread_min_max'][:, 0] == -np.inf), (
        'As MultiNest does not currently perform dynamic nested sampling, all '
        'threads should start by sampling the whole prior.')
    ns_run['output'] = {}
    ns_run['output']['file_root'] = file_root
    ns_run['output']['base_dir'] = base_dir
    return ns_run