def _terminate_procs(procs):
    """
    Terminate all processes in the process dictionary
    """
    logging.warn("Stopping all remaining processes")
    for proc, g in procs.values():
        logging.debug("[%s] SIGTERM", proc.pid)
        try:
            proc.terminate()
        except OSError as e:
            # we don't care if the process we tried to kill didn't exist.
            if e.errno != errno.ESRCH:
                raise
    sys.exit(1)