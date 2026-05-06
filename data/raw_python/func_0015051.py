def print_commands():
    """Prints all commands available from Log with their
    description.
    """
    dummy_log_file = Log()
    commands = Log.commands()
    commands.sort()

    for cmd in commands:
        cmd = getattr(dummy_log_file, 'cmd_{0}'.format(cmd))
        description = cmd.__doc__
        if description:
            description = re.sub(r'\n\s+', ' ', description)
            description = description.strip()

        print('{0}: {1}\n'.format(cmd.__name__, description))