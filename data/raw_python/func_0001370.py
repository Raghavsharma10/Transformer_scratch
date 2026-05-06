def signal(*args, **kwargs):
    from .core import Signal
    """A signal decorator designed to work both in the simpler way, like:

    .. code:: python

      @signal
      def validation_function(arg1, ...):
          '''Some doc'''

    and also as a double-called decorator, like

    .. code:: python

      @signal(SignalOptions.EXEC_CONCURRENT)
      def validation_function(arg1, ...):
          '''Some doc'''

    """
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return Signal(fvalidation=args[0])
    else:
        sig = Signal(*args, **kwargs)
        def wrapper(fvalidation):
            sig._set_fvalidation(fvalidation)
            return sig
        return wrapper