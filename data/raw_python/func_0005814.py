def make_rpc_call(func_name, args=None, remote=None):
    """Performs an RPC function call (local or remote) with the given arguments.

    :param str|unicode func_name: RPC function name to call.

    :param Iterable args: Function arguments.

    :param str|unicode remote:

    :rtype: bytes|str

    :raises ValueError: If unable to call RPC function.

    """
    args = args or []
    args = [encode(str(arg)) for arg in args]

    if remote:
        result = uwsgi.rpc(remote, func_name, *args)
    else:
        result = uwsgi.call(func_name, *args)

    return decode(result)