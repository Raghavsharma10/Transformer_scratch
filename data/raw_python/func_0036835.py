def wait_on_any(*events, **kwargs):
    """
    Helper method for waiting for any of the given threading events to be
    set.

    The standard threading lib doesn't include any mechanism for waiting on
    more than one event at a time so we have to monkey patch the events
    so that their `set()` and `clear()` methods fire a callback we can use
    to determine how a composite event should react.
    """
    timeout = kwargs.get("timeout")
    composite_event = threading.Event()

    if any([event.is_set() for event in events]):
        return

    def on_change():
        if any([event.is_set() for event in events]):
            composite_event.set()
        else:
            composite_event.clear()

    def patch(original):

        def patched():
            original()
            on_change()

        return patched

    for event in events:
        event.set = patch(event.set)
        event.clear = patch(event.clear)

    wait_on_event(composite_event, timeout=timeout)