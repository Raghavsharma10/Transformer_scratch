def run_subprocess(command, return_code=False, **kwargs):
    """Run command using subprocess.Popen

    Run command and wait for command to complete. If the return code was zero
    then return, otherwise raise CalledProcessError.
    By default, this will also add stdout= and stderr=subproces.PIPE
    to the call to Popen to suppress printing to the terminal.

    Parameters
    ----------
    command : list of str
        Command to run as subprocess (see subprocess.Popen documentation).
    return_code : bool
        If True, the returncode will be returned, and no error checking
        will be performed (so this function should always return without
        error).
    **kwargs : dict
        Additional kwargs to pass to ``subprocess.Popen``.

    Returns
    -------
    stdout : str
        Stdout returned by the process.
    stderr : str
        Stderr returned by the process.
    code : int
        The command exit code. Only returned if ``return_code`` is True.
    """
    # code adapted with permission from mne-python
    use_kwargs = dict(stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    use_kwargs.update(kwargs)

    p = subprocess.Popen(command, **use_kwargs)
    output = p.communicate()

    # communicate() may return bytes, str, or None depending on the kwargs
    # passed to Popen(). Convert all to unicode str:
    output = ['' if s is None else s for s in output]
    output = [s.decode('utf-8') if isinstance(s, bytes) else s for s in output]
    output = tuple(output)

    if not return_code and p.returncode:
        print(output[0])
        print(output[1])
        err_fun = subprocess.CalledProcessError.__init__
        if 'output' in inspect.getargspec(err_fun).args:
            raise subprocess.CalledProcessError(p.returncode, command, output)
        else:
            raise subprocess.CalledProcessError(p.returncode, command)
    if return_code:
        output = output + (p.returncode,)
    return output