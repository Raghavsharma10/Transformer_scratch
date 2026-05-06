def migrate(*argv) -> bool:
    """
    Runs Django migrate command.

    :return: always ``True``
    """
    wf('Applying migrations... ', False)
    execute_from_command_line(['./manage.py', 'migrate'] + list(argv))
    wf('[+]\n')
    return True