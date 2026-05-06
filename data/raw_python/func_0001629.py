def start_service(addr, n, authenticator):
    """ Start a service """

    s = Subscriber(addr, authenticator=authenticator)

    def do_something(line):
        pass

    s.subscribe('test', do_something)

    started = time.time()
    for _ in range(n):
        s.process()
    s.socket.close()
    duration = time.time() - started

    print('Subscriber service stats:')
    util.print_stats(n, duration)
    return