def execute(command, *args, **kwargs):
    """Execute a command with arguments and wait for output.

    Arguments should not be quoted!

    Keyword arguments:
        env (dict): Dictionary of additional environment variables.
        wait (bool): Wait for the process to finish.

    Example::

        >>> code = 'import sys;sys.stdout.write('out');sys.exit(0)'
        >>> status, out, err = execute('python', '-c', code)
        >>> print('status: %s, output: %s, error: %s' % (status, out, err))
        status: 0, output: out, error:
        >>> code = 'import sys;sys.stderr.write('out');sys.exit(1)'
        >>> status, out, err = execute('python', '-c', code)
        >>> print('status: %s, output: %s, error: %s' % (status, out, err))
        status: 1, output: , error: err
    """
    wait = kwargs.pop("wait", True)
    process = Process(command, args, env=kwargs.pop("env", None))
    process.start()
    if not wait:
        return process
    process.wait()
    return process.exit_code, process.read(), process.eread()