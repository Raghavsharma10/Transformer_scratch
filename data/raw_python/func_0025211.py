def setup(cmd_args, suppress_output=False):
    """ Call a setup.py command or list of commands

    >>> result = setup('--name', suppress_output=True)
    >>> result.exitval
    0
    >>> result = setup('notreal')
    >>> result.exitval
    1
    """
    if not funcy.is_list(cmd_args) and not funcy.is_tuple(cmd_args):
        cmd_args = shlex.split(cmd_args)
    cmd_args = [sys.executable, 'setup.py'] + [x for x in cmd_args]
    return call(cmd_args, suppress_output=suppress_output)