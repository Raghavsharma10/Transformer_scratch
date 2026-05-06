def run(cmd,
        capture=False,
        shell=True,
        env=None,
        exit_on_error=None,
        never_pretend=False):
    # type: (str, bool, bool, Dict[str, str], bool) -> ExecResult
    """ Run a shell command.

    Args:
        cmd (str):
            The shell command to execute.
        shell (bool):
            Same as in `subprocess.Popen`.
        capture (bool):
            If set to True, it will capture the standard input/error instead of
            just piping it to the caller stdout/stderr.
        env (dict[str, str]):
            The subprocess environment variables.
        exit_on_error (bool):
            If set to **True**, on failure it will call `sys.exit` with the
            return code for the executed command.
        never_pretend (bool):
            If set to **True** the command will always be executed, even if
            context.get('pretend') is set to True. If set to **False** or not
            given, if the `pretend` context value is **True**, this function
            will only print the command it would execute and then return
            a fake result.

    Returns:
        ExecResult: The execution result containing the return code and output
        (if capture was set to *True*).
    """
    if context.get('pretend', False) and not never_pretend:
        cprint('<90>{}', cmd)
        return ExecResult(
            cmd,
            0,              # retcode
            '',             # stdout
            '',             # stderr
            True,           # succeeded
            False,          # failed
        )

    if context.get('verbose', 0) > 2:
        cprint('<90>{}', cmd)

    options = {
        'bufsize': 1,       # line buffered
        'shell': shell
    }

    if exit_on_error is None:
        exit_on_error = not capture

    if capture:
        options.update({
            'stdout': subprocess.PIPE,
            'stderr': subprocess.PIPE,
        })

    if env is not None:
        options['env'] = dict(os.environ)
        options['env'].update(env)

    p = subprocess.Popen(cmd, **options)
    stdout, stderr = p.communicate()

    try:
        if stdout is not None:
            stdout = stdout.decode('utf-8')

        if stderr is not None:
            stderr = stderr.decode('utf-8')
    except AttributeError:
        # 'str' has no attribute 'decode'
        pass

    if exit_on_error and p.returncode != 0:
        sys.exit(p.returncode)

    return ExecResult(
        cmd,
        p.returncode,
        stdout,
        stderr,
        p.returncode == 0,
        p.returncode != 0
    )