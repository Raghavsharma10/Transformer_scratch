def _validate_options(options, service_name, add_error):
    """Lazily validate the options, ensuring that they are a dict.

    Use the given add_error callable to register validation error.
    """
    if options is None:
        return
    if not isdict(options):
        add_error('service {} has malformed options'.format(service_name))