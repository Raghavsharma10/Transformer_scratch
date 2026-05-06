def remove_signals_listeners(instance):
    """
    utility function that disconnects all listeners from all signals on an
    object
    """
    if hasattr(instance, "__listeners__"):
        for listener in list(instance.__listeners__):
            for signal in instance.__listeners__[listener]:
                signal.disconnect(listener)