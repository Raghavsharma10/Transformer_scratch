def profileit(func):
    """
    Decorator straight up stolen from stackoverflow
    """
    def wrapper(*args, **kwargs):
        datafn = func.__name__ + ".profile" # Name the data file sensibly
        prof = cProfile.Profile()
        prof.enable()
        retval = prof.runcall(func, *args, **kwargs)
        prof.disable()
        stats = pstats.Stats(prof)
        try:
            stats.sort_stats('cumtime').print_stats()
        except KeyError:
            pass  # breaks in python 2.6
        return retval

    return wrapper