def event_handler(event_name):
    """
    Decorator for designating a handler for an event type. ``event_name`` must be a string
    representing the name of the event type.

    The decorated function must accept a parameter: the body of the received event,
    which will be a Python object that can be encoded as a JSON (dict, list, str, int,
    bool, float or None)

    :param event_name: The name of the event that will be handled. Only one handler per
                       event name is supported by the same microservice.
    """

    def wrapper(func):
        func._event_handler = True
        func._handled_event = event_name
        return func

    return wrapper