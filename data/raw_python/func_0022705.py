def _enable_profiling():
    """ Start profiling and register callback to print stats when the program
    exits.
    """
    import cProfile
    import atexit
    global _profiler
    _profiler = cProfile.Profile()
    _profiler.enable()
    atexit.register(_profile_atexit)