def run(*popenargs, **kwargs):
    """Run command with arguments and return a `CompletedProcess` instance.

    The returned instance will have attributes args, returncode, stdout and
    stderr.

    By default, stdout and stderr are not captured, and those attributes
    will be None. Pass stdout=PIPE and/or stderr=PIPE in order to capture them.

    If `check` is True and the exit code was non-zero, it raises a
    CalledProcessError. The CalledProcessError object will have the return code
    in the returncode attribute, and output & stderr attributes if those
    streams were captured.

    If `timeout` is given, and the process takes too long, a TimeoutExpired
    exception will be raised, if timeout is supported in the underlying Popen
    implementation (e.g. Python >= 3.2, or an available subprocess32 package).

    There is an optional argument `input`, allowing you to
    pass a string to the subprocess's stdin.  If you use this argument
    you may not also use the Popen constructor's `stdin` argument, as
    it will be used internally.

    The other arguments are the same as for the Popen constructor.

    If universal_newlines=True is passed, the `input` argument must be a
    string and stdout/stderr in the returned object will be strings rather than
    bytes.
    """
    stdin = kwargs.pop('input', None)
    timeout = kwargs.pop('timeout', None)
    check = kwargs.pop('check', False)
    if stdin is not None:
        if 'stdin' in kwargs:
            raise ValueError('stdin and input arguments may not both be used.')
        kwargs['stdin'] = PIPE

    process = Popen(*popenargs, **kwargs)
    try:
        if __timeout__:
            stdout, stderr = process.communicate(stdin, timeout=timeout)
        else:
            stdout, stderr = process.communicate(stdin)
    except TimeoutExpired:
        # this will never happen if __timeout__ is False
        process.kill()
        stdout, stderr = process.communicate()
        # pylint: disable=no-member
        raise _TimeoutExpired(process.args, timeout, output=stdout,
                              stderr=stderr)
    except:
        process.kill()
        process.wait()
        raise
    retcode = process.poll()
    if check and retcode:
        raise CalledProcessError(retcode, popenargs,
                                 output=stdout, stderr=stderr)
    return CompletedProcess(popenargs, retcode, stdout, stderr)