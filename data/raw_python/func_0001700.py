def bench(client, n):
    """ Benchmark n requests """
    pairs = [(x, x + 1) for x in range(n)]

    started = time.time()
    for pair in pairs:
        res, err = client.call('add', *pair)
        # assert err is None
    duration = time.time() - started
    print('Client stats:')
    util.print_stats(n, duration)