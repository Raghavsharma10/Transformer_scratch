def repo_name(msg):
    """ Compat util to get the repo name from a message. """
    try:
        # git messages look like this now
        path = msg['msg']['commit']['path']
        project = path.split('.git')[0][9:]
    except KeyError:
        # they used to look like this, though
        project = msg['msg']['commit']['repo']

    return project