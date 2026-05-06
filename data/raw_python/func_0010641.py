def _run_as(user, group):
        """ Function wrapper that sets the user and group for the process """
        def wrapper():
            if user is not None:
                os.setuid(user)
            if group is not None:
                os.setgid(group)
        return wrapper