def io_size_kb(prev, curr, counters):
    """ calculate the io size based on bandwidth and throughput

    formula: average_io_size = bandwidth / throughput
    :param prev: prev resource, not used
    :param curr: current resource
    :param counters: two stats, bandwidth in MB and throughput count
    :return: value, NaN if invalid
    """
    bw_stats, io_stats = counters
    size_mb = div(getattr(curr, bw_stats), getattr(curr, io_stats))
    return mul(size_mb, 1024)