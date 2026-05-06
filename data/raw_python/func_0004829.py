def _detect_notebook() -> bool:
    """Detect if code is running in a Jupyter Notebook.

    This isn't 100% correct but seems good enough

    Returns
    -------
    bool
        True if it detects this is a notebook, otherwise False.

    """
    try:
        from IPython import get_ipython
        from ipykernel import zmqshell
    except ImportError:
        return False
    kernel = get_ipython()
    try:
        from spyder.utils.ipython.spyder_kernel import SpyderKernel

        if isinstance(kernel.kernel, SpyderKernel):
            return False
    except (ImportError, AttributeError):
        pass
    return isinstance(kernel, zmqshell.ZMQInteractiveShell)