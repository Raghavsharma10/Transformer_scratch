def get_git_changeset(filename=None):
    """Returns a numeric identifier of the latest git changeset.

    The result is the UTC timestamp of the changeset in YYYYMMDDHHMMSS format.
    This value isn't guaranteed to be unique, but collisions are very unlikely,
    so it's sufficient for generating the development version numbers.
    """
    dirname = os.path.dirname(filename or __file__)
    git_show = sh('git show --pretty=format:%ct --quiet HEAD',
                  cwd=dirname)
    timestamp = git_show.partition('\n')[0]
    try:
        timestamp = datetime.datetime.utcfromtimestamp(int(timestamp))
    except ValueError:
        return None
    return timestamp.strftime('%Y%m%d%H%M%S')