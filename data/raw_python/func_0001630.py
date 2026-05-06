def bench(client, n):
    """ Benchmark n requests """
    items = list(range(n))

    # Time client publish operations
    # ------------------------------
    started = time.time()
    for i in items:
        client.publish('test', i)
    duration = time.time() - started

    print('Publisher client stats:')
    util.print_stats(n, duration)