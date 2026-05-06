def fixed_interval_scheduler(interval):
    """
    A scheduler that ticks at fixed intervals of "interval" seconds
    """
    start = time.time()
    next_tick = start

    while True:
        next_tick += interval
        yield next_tick