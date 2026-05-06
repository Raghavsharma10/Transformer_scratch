def _blocking(lock, state_dict, event, timeout=None):
    """
    A contextmanager that clears `state_dict` and `event`, yields, and waits
    for the event to be set. Clearing an yielding are done within `lock`.

    Used for blocking request/response semantics on the request side, as in:

        with _blocking(lock, state, event):
            send_request()

    The response side would then do something like:

        with lock:
            state['data'] = '...'
            event.set()
    """
    with lock:
        state_dict.clear()
        event.clear()
        yield
    event.wait(timeout)