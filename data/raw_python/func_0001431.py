def git_tags() -> str:
    """
    Calls ``git tag -l --sort=-v:refname`` (sorts output) and returns the
    output as a UTF-8 encoded string. Raises a NoGitTagsException if the
    repository doesn't contain any Git tags.
    """
    try:
        subprocess.check_call(['git', 'fetch', '--tags'])
    except CalledProcessError:
        pass

    cmd = ['git', 'tag', '--list', '--sort=-v:refname']
    rv = subprocess.check_output(cmd).decode('utf-8')

    if rv == '':
        raise NoGitTagsException('No Git tags are present in current repo.')

    return rv