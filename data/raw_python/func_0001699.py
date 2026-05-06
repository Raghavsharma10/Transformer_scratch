def start_service(addr, n):
    """ Start a service """
    s = Service(addr)
    s.register('add', lambda x, y: x + y)

    started = time.time()
    for _ in range(n):
        s.process()
    duration = time.time() - started

    time.sleep(0.1)
    print('Service stats:')
    util.print_stats(n, duration)
    return