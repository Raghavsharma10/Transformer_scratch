def cmd(command, ignore_stderr=False, raise_on_return=False, timeout=None,
        encoding="utf-8"):
    """ Run a shell command and have it automatically decoded and printed

    :param command: Command to run as str
    :param ignore_stderr: To not print stderr
    :param raise_on_return: Run CompletedProcess.check_returncode()
    :param timeout: timeout to pass to communicate if python 3
    :param encoding: How the output should be decoded
    """
    result = run(command, timeout=timeout, shell=True)
    if raise_on_return:
        result.check_returncode()
    print(result.stdout.decode(encoding))
    if not ignore_stderr and result.stderr:
        print(result.stderr.decode(encoding))