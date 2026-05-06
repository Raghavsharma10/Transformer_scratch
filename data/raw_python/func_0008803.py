def _sf2(args):
    """
    A shallow wrapper for sigma_filter.

    Parameters
    ----------
    args : list
        A list of arguments for sigma_filter

    Returns
    -------
    None
    """
    # an easier to debug traceback when multiprocessing
    # thanks to https://stackoverflow.com/a/16618842/1710603
    try:
        return sigma_filter(*args)
    except:
        import traceback
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))