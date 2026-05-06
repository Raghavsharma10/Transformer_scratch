def git_commits_since_last_tag(last_tag: str) -> dict:
    """
    :last_tag: The Git tag that should serve as the starting point for the
    commit log lookup.

    Calls ``git log <last_tag>.. --format='%H %s'`` and returns the output as a
    dict of hash-message pairs.
    """
    try:
        cmd = ['git', 'log', last_tag + '..', "--format='%H %s'"]
        commit_log = subprocess.check_output(cmd).decode('utf-8')
    except CalledProcessError:
        raise GitTagDoesNotExistError('No such tag:', last_tag)

    if not commit_log:
        raise NoGitCommitSinceLastTagException('No commits since last tag.')

    pattern = re.compile(r'([a-f0-9]{40})\ (.*)')

    rv = {}
    for line in commit_log.split('\n'):
        match = pattern.search(line)
        if match:
            commit_hash = match.group(1)
            commit_msg = match.group(2)
            rv[commit_hash] = commit_msg

    return rv