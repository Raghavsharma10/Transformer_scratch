def remove(id_):
    """
    Remove the callback and its schedule
    """
    with LOCK:
        thread = REGISTRY.pop(id_, None)
        if thread is not None:
            thread.cancel()

    return thread