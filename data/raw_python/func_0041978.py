def _verbose_message(message, *args, **kwargs):
    """Print the message to stderr if -v/PYTHONVERBOSE is turned on."""
    verbosity = kwargs.pop('verbosity', 1)
    if sys.flags.verbose >= verbosity:
        if not message.startswith(('#', 'import ')):
            message = '# ' + message
        print(message.format(*args), file=sys.stderr)