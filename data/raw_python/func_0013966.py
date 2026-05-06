def duel_command(f):
    """ indicate it's a command need to be called on both SP

    :param f: function that returns the command in list
    :return: command execution result on both sps (tuple of 2)
    """

    @functools.wraps(f)
    def func_wrapper(self, *argv, **kwargs):
        commands = _get_commands(f, self, *argv, **kwargs)
        return self.execute_dual(commands)

    return func_wrapper