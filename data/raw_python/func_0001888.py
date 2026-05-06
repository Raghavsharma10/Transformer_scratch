def bench(client, n):
    """ Benchmark n requests """
    items = list(range(n))

    # Time client publish operations
    # ------------------------------
    started = time.time()
    msg = b'x'
    for i in items:
        client.socket.send(msg)
        res = client.socket.recv()
        assert msg == res
    duration = time.time() - started

    print('Raw REQ client stats:')
    util.print_stats(n, duration)