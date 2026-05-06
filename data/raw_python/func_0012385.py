def process_polychord_run(file_root, base_dir, process_stats_file=True,
                          **kwargs):
    """Loads data from a PolyChord run into the nestcheck dictionary format for
    analysis.

    N.B. producing required output file containing information about the
    iso-likelihood contours within which points were sampled (where they were
    "born") requies PolyChord version v1.13 or later and the setting
    write_dead=True.

    Parameters
    ----------
    file_root: str
        Root for run output file names (PolyChord file_root setting).
    base_dir: str
        Directory containing data (PolyChord base_dir setting).
    process_stats_file: bool, optional
        Should PolyChord's <root>.stats file be processed? Set to False if you
        don't have the <root>.stats file (such as if PolyChord was run with
        write_stats=False).
    kwargs: dict, optional
        Options passed to ns_run_utils.check_ns_run.

    Returns
    -------
    ns_run: dict
        Nested sampling run dict (see the module docstring for more details).
    """
    # N.B. PolyChord dead points files also contains remaining live points at
    # termination
    samples = np.loadtxt(os.path.join(base_dir, file_root) + '_dead-birth.txt')
    ns_run = process_samples_array(samples, **kwargs)
    ns_run['output'] = {'base_dir': base_dir, 'file_root': file_root}
    if process_stats_file:
        try:
            ns_run['output'] = process_polychord_stats(file_root, base_dir)
        except (OSError, IOError, ValueError) as err:
            warnings.warn(
                ('process_polychord_stats raised {} processing {}.stats file. '
                 ' Proceeding without stats.').format(
                     type(err).__name__, os.path.join(base_dir, file_root)),
                UserWarning)
    return ns_run