def lock_access(repository_path, callback):
    """ Synchronise access to the user file between processes, this specifies
    which user is allowed write access at the current time """

    with open(cpjoin(repository_path, 'lock_file'), 'w') as fd:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            returned = callback()
            fcntl.flock(fd, fcntl.LOCK_UN)
            return returned
        except IOError:
            return fail(lock_fail_msg)