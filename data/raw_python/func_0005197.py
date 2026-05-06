def commit_branches(sha1):
    # type: (str) -> List[str]
    """ Get the name of the branches that this commit belongs to. """
    cmd = 'git branch --contains {}'.format(sha1)
    return shell.run(
        cmd,
        capture=True,
        never_pretend=True
    ).stdout.strip().split()