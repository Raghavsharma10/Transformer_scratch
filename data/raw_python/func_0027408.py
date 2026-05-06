def zthread_fork(ctx, func, *args, **kwargs):
    """
    Create an attached thread. An attached thread gets a ctx and a PAIR
    pipe back to its parent. It must monitor its pipe, and exit if the
    pipe becomes unreadable. Returns pipe, or NULL if there was an error.
    """
    a = ctx.socket(zmq.PAIR)
    a.setsockopt(zmq.LINGER, 0)
    a.setsockopt(zmq.RCVHWM, 100)
    a.setsockopt(zmq.SNDHWM, 100)
    a.setsockopt(zmq.SNDTIMEO, 5000)
    a.setsockopt(zmq.RCVTIMEO, 5000)
    b = ctx.socket(zmq.PAIR)
    b.setsockopt(zmq.LINGER, 0)
    b.setsockopt(zmq.RCVHWM, 100)
    b.setsockopt(zmq.SNDHWM, 100)
    b.setsockopt(zmq.SNDTIMEO, 5000)
    a.setsockopt(zmq.RCVTIMEO, 5000)
    iface = "inproc://%s" % binascii.hexlify(os.urandom(8))
    a.bind(iface)
    b.connect(iface)

    thread = threading.Thread(target=func, args=((ctx, b) + args), kwargs=kwargs)
    thread.daemon = False
    thread.start()

    return a