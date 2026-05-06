def check_cores(cores):
    """
    Determine how many cores we are able to use.
    Return 1 if we are not able to make a queue via pprocess.

    Parameters
    ----------
    cores : int
        The number of cores that are requested.

    Returns
    -------
    cores : int
        The number of cores available.

    """
    cores = min(multiprocessing.cpu_count(), cores)
    if six.PY3:
        log = logging.getLogger("Aegean")
        log.info("Multi-cores not supported in python 3+, using one core")
        return 1
    try:
        queue = pprocess.Queue(limit=cores, reuse=1)
    except:  # TODO: figure out what error is being thrown
        cores = 1
    else:
        try:
            _ = queue.manage(pprocess.MakeReusable(fix_shape))
        except:
            cores = 1
    return cores