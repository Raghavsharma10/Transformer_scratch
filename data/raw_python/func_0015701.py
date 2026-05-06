def signal_handler_block(obj, handler_id):
    """Blocks the signal handler from being invoked until
    handler_unblock() is called.

    :param GObject.Object obj:
        Object instance to block handlers for.
    :param int handler_id:
        Id of signal to block.
    :returns:
        A context manager which optionally can be used to
        automatically unblock the handler:

    .. code-block:: python

        with GObject.signal_handler_block(obj, id):
            pass
    """
    GObjectModule.signal_handler_block(obj, handler_id)
    return _HandlerBlockManager(obj, handler_id)