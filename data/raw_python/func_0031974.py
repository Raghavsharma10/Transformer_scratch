def register(callback, schedule, tag=None):
    """
    Register a callback which will be called at scheduled intervals with
    the metrics that have the given tag (or all the metrics if None).
    Return an identifier which can be used to access the registered callback later.
    """

    try:
        iter(schedule)
    except TypeError:
        raise TypeError("{} is not iterable".format(schedule))

    if not callable(callback):
        raise TypeError("{} is not callable".format(callback))

    thread = Timer(schedule, callback, tag)

    id_ = str(uuid.uuid4())
    with LOCK:
        REGISTRY[id_] = thread

    thread.start()

    return id_