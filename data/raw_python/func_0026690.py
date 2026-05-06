def _call_command_in_repo(comm, repo, log, fail=False, log_flag=True):
    """Use `subprocess` to call a command in a certain (repo) directory.

    Logs the output (both `stderr` and `stdout`) to the log, and checks the
    return codes to make sure they're valid.  Raises error if not.

    Raises
    ------
    exception `subprocess.CalledProcessError`: if the command fails

    """
    if log_flag:
        log.debug("Running '{}'.".format(" ".join(comm)))
    process = subprocess.Popen(
        comm, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = process.communicate()
    if stderr is not None:
        err_msg = stderr.decode('ascii').strip().splitlines()
        for em in err_msg:
            log.error(em)
    if stdout is not None:
        out_msg = stdout.decode('ascii').strip().splitlines()
        for om in out_msg:
            log.warning(om)
    # Raises an error if the command failed.
    if fail:
        if process.returncode:
            raise subprocess.CalledProcessError
    return