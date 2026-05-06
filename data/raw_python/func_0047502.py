def aggregate_registry_timers():
    """Returns a list of aggregate timing information for registered timers.

    Each element is a 3-tuple of

        - timer description
        - aggregate elapsed time
        - number of calls

    The list is sorted by the first start time of each aggregate timer.

    """
    import itertools

    timers = sorted(shared_registry.values(), key=lambda t: t.desc)
    aggregate_timers = []
    for k, g in itertools.groupby(timers, key=lambda t: t.desc):
        group = list(g)
        num_calls = len(group)
        total_elapsed_ms = sum(t.elapsed_time_ms for t in group)
        first_start_time = min(t.start_time for t in group)
        # We'll use the first start time as a sort key.
        aggregate_timers.append(
            (first_start_time, (k, total_elapsed_ms, num_calls)))

    aggregate_timers.sort()
    return zip(*aggregate_timers)[1]