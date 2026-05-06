def _get_cmd(command, arguments):
    """Merge command with arguments."""
    if arguments is None:
        arguments = []
    if command.endswith(".py") or command.endswith(".pyw"):
        return [sys.executable, command] + list(arguments)
    else:
        return [command] + list(arguments)