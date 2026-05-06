def get_sha(path=None, log=None, short=False, timeout=None):
    """Use `git rev-parse HEAD <REPO>` to get current SHA.
    """
    # git_command = "git rev-parse HEAD {}".format(repo_name).split()
    # git_command = "git rev-parse HEAD".split()
    git_command = ["git", "rev-parse"]
    if short:
        git_command.append("--short")
    git_command.append("HEAD")

    kwargs = {}
    if path is not None:
        kwargs['cwd'] = path
    if timeout is not None:
        kwargs['timeout'] = timeout

    if log is not None:
        log.debug("{} {}".format(git_command, str(kwargs)))

    sha = subprocess.check_output(git_command, **kwargs)
    try:
        sha = sha.decode('ascii').strip()
    except:
        if log is not None:
            log.debug("decode of '{}' failed".format(sha))

    return sha