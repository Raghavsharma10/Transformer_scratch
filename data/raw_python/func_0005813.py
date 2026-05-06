def register_rpc(name=None):
    """Decorator. Allows registering a function for RPC.

    * http://uwsgi.readthedocs.io/en/latest/RPC.html

    Example:

        .. code-block:: python

            @register_rpc()
            def expose_me():
                do()


    :param str|unicode name: RPC function name to associate
        with decorated function.

    :rtype: callable
    """
    def wrapper(func):
        func_name = func.__name__
        rpc_name = name or func_name

        uwsgi.register_rpc(rpc_name, func)

        _LOG.debug("Registering '%s' for RPC under '%s' alias ...", func_name, rpc_name)

        return func

    return wrapper