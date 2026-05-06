def _run_command(command, targets, options):
    # type: (str, List[str], List[str]) -> bool
    """Runs `command` + `targets` + `options` in a
    subprocess and returns a boolean determined by the
    process return code.

    >>> result = run_command('pylint', ['foo.py', 'some_module'], ['-E'])
    >>> result
    True

    :param command: str
    :param targets: List[str]
    :param options: List[str]
    :return: bool
    """
    print('{0}: targets={1} options={2}'.format(command, targets, options))
    cmd = [command] + targets + options
    process = Popen(cmd)
    process.wait()

    return bool(process.returncode)