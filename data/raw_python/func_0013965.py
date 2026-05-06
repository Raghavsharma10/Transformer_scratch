def command(f):
    """ indicate it's a command of naviseccli

    :param f: function that returns the command in list
    :return: command execution result
    """

    @functools.wraps(f)
    def func_wrapper(self, *argv, **kwargs):
        if 'ip' in kwargs:
            ip = kwargs['ip']
            del kwargs['ip']
        else:
            ip = None

        commands = _get_commands(f, self, *argv, **kwargs)
        return self.execute(commands, ip=ip)

    return func_wrapper