def _in_qtconsole() -> bool:
    """
    A small utility function which determines if we're running in QTConsole's context.
    """
    try:
        from IPython import get_ipython
        try:
            from ipykernel.zmqshell import ZMQInteractiveShell
            shell_object = ZMQInteractiveShell
        except ImportError:
            from IPython.kernel.zmq import zmqshell
            shell_object = zmqshell.ZMQInteractiveShell
        return isinstance(get_ipython(), shell_object)
    except Exception:
        return False