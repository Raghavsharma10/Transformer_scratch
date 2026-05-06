def start_service(addr, n):
    """ Start a service """
    s = Service(addr)

    started = time.time()
    for _ in range(n):
        msg = s.socket.recv()
        s.socket.send(msg)
    s.socket.close()
    duration = time.time() - started

    print('Raw REP service stats:')
    util.print_stats(n, duration)
    return