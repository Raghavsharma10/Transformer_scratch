def commit_author(sha1=''):
    # type: (str) -> Author
    """ Return the author of the given commit.

    Args:
        sha1 (str):
            The sha1 of the commit to query. If not given, it will return the
            sha1 for the current commit.
    Returns:
        Author: A named tuple ``(name, email)`` with the commit author details.
    """
    with conf.within_proj_dir():
        cmd = 'git show -s --format="%an||%ae" {}'.format(sha1)
        result = shell.run(
            cmd,
            capture=True,
            never_pretend=True
        ).stdout
        name, email = result.split('||')
        return Author(name, email)