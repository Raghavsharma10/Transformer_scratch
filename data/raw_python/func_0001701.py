def start_service(addr, n):
    """ Start a service """

    s = Subscriber(addr)
    s.socket.set_string_option(nanomsg.SUB, nanomsg.SUB_SUBSCRIBE, 'test')

    started = time.time()
    for _ in range(n):
        msg = s.socket.recv()
    s.socket.close()
    duration = time.time() - started

    print('Raw SUB service stats:')
    util.print_stats(n, duration)
    return