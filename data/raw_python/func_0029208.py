def get_complete(command):
    """
    Get the Cmd complete function for the click command
    :param command: The click Command object
    :return: the complete_* method for Cmd
    :rtype: function
    """

    assert isinstance(command, click.Command)

    def complete_(self, text, line, begidx, endidx):  # pylint: disable=unused-argument
        # Parse the args
        args = shlex.split(line[:begidx])
        # Strip of the first item which is the name of the command
        args = args[1:]

        # Then pass them on to the get_choices method that click uses for completion
        return list(get_choices(command, command.name, args, text))

    complete_.__name__ = 'complete_%s' % command.name
    return complete_