def run(command=None, *arguments):
    """
    Run the given command.

    Parameters:
    :param command: A string describing a command.
    :param arguments: A list of strings describing arguments to the command.
    """

    if command is None:
        sys.exit('django-shortcuts: No argument was supplied, please specify one.')

    if command in ALIASES:
        command = ALIASES[command]

    if command == 'startproject':
        return call('django-admin.py startproject %s' % ' '.join(arguments), shell=True)

    script_path = os.getcwd()
    while not os.path.exists(os.path.join(script_path, 'manage.py')):
        base_dir = os.path.dirname(script_path)
        if base_dir != script_path:
            script_path = base_dir
        else:
            sys.exit('django-shortcuts: No \'manage.py\' script found in this directory or its parents.')

    return call('%(python)s %(script_path)s %(command)s %(arguments)s' % {
        'python': sys.executable,
        'script_path': os.path.join(script_path, 'manage.py'),
        'command': command or '',
        'arguments': ' '.join(arguments)
    }, shell=True)