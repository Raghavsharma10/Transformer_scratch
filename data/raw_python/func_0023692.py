def profiler():
    '''Profile the block'''
    import cProfile
    import pstats
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()
    ps = pstats.Stats(pr).sort_stats('tottime')
    ps.print_stats()