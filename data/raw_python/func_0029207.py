def get_help(command):
    """
    Get the Cmd help function from the click command
    :param command: The click Command object
    :return: the help_* method for Cmd
    :rtype: function
    """
    assert isinstance(command, click.Command)

    def help_(self):  # pylint: disable=unused-argument
        extra = {}
        for key, value in command.context_settings.items():
            if key not in extra:
                extra[key] = value

        # Print click's help message
        with click.Context(command, info_name=command.name, parent=self.ctx, **extra) as ctx:
            click.echo(ctx.get_help(), color=ctx.color)

    help_.__name__ = 'help_%s' % command.name
    return help_