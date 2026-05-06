def nas_command(f):
    """ indicate it's a command of nas command run with ssh

    :param f: function that returns the command in list
    :return: command execution result
    """

    @functools.wraps(f)
    def func_wrapper(self, *argv, **kwargs):
        commands = f(self, *argv, **kwargs)
        return self.ssh_execute(['env', 'NAS_DB=/nas'] + commands)

    return func_wrapper