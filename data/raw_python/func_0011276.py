def command(name=None):
    """A decorator to register a subcommand with the global `Subcommands` instance.
    """
    def decorator(f):
        _commands.append((name, f))
        return f
    return decorator