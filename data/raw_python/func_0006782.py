def dir_exists(location, use_sudo=False):
    """Tells if there is a remote directory at the given location."""
    with settings(hide('running', 'stdout', 'stderr'), warn_only=True):
        if use_sudo:
            # convert return code 0 to True
            return not bool(sudo('test -d %s' % (location)).return_code)
        else:
            return not bool(run('test -d %s' % (location)).return_code)