def sudo_run(c, command):
    """
    Run some command under Travis-oriented sudo subshell/virtualenv.

    :param str command:
        Command string to run, e.g. ``inv coverage``, ``inv integration``, etc.
        (Does not necessarily need to be an Invoke task, but...)
    """
    # NOTE: explicit shell wrapper because sourcing the venv works best here;
    # test tasks currently use their own subshell to call e.g. 'pytest --blah',
    # so the tactic of '$VIRTUAL_ENV/bin/inv coverage' doesn't help - only that
    # intermediate process knows about the venv!
    cmd = "source $VIRTUAL_ENV/bin/activate && {}".format(command)
    c.sudo('bash -c "{0}"'.format(cmd), user=c.travis.sudo.user)