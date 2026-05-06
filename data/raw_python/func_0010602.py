def event_stream(app, *, filter_by_prefix=None):
    """ Generator function that returns celery events.

    This function turns the callback based celery event handling into a generator.

    Args:
        app: Reference to a celery application object.
        filter_by_prefix (str): If not None, only allow events that have a type that
                                 starts with this prefix to yield an generator event.

    Returns:
        generator: A generator that returns celery events.

    """
    q = Queue()

    def handle_event(event):
        if filter_by_prefix is None or\
                (filter_by_prefix is not None and
                 event['type'].startswith(filter_by_prefix)):
            q.put(event)

    def receive_events():
        with app.connection() as connection:
            recv = app.events.Receiver(connection, handlers={
                '*': handle_event
            })

            recv.capture(limit=None, timeout=None, wakeup=True)

    t = threading.Thread(target=receive_events)
    t.start()

    while True:
        yield q.get(block=True)