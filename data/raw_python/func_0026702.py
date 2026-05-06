def log_memory(log, pref=None, lvl=logging.DEBUG, raise_flag=True):
    """Log the current memory usage.
    """
    import os
    import sys
    cyc_str = ""
    KB = 1024.0
    if pref is not None:
        cyc_str += "{}: ".format(pref)

    # Linux returns units in Bytes; OSX in kilobytes
    UNIT = KB*KB if sys.platform == 'darwin' else KB

    good = False
    # Use the `resource` module to check the maximum memory usage of this process
    try:
        import resource
        max_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        max_child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        _str = "RSS Max Self: {:7.2f} [MB], Child: {:7.2f} [MB]".format(
            max_self/UNIT, max_child/UNIT)
        cyc_str += _str
    except Exception as err:
        log.log(lvl, "resource.getrusage failed.  '{}'".format(str(err)))
        if raise_flag:
            raise
    else:
        good = True

    # Use the `psutil` module to check the current memory/cpu usage of this process
    try:
        import psutil
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss
        cpu_perc = process.cpu_percent()
        mem_perc = process.memory_percent()
        num_thr = process.num_threads()
        _str = "; RSS: {:7.2f} [MB], {:7.2f}%; Threads: {:3d}, CPU: {:7.2f}%".format(
            rss/UNIT, mem_perc, num_thr, cpu_perc)
        cyc_str += _str
    except Exception as err:
        log.log(lvl, "psutil.Process failed.  '{}'".format(str(err)))
        if raise_flag:
            raise
    else:
        good = True

    if good:
        log.log(lvl, cyc_str)

    return