def init(init_type='plaintext_tcp', *args, **kwargs):
    """
    Create the module instance of the GraphiteClient.
    """
    global _module_instance
    reset()

    validate_init_types = ['plaintext_tcp', 'plaintext', 'pickle_tcp',
                           'pickle', 'plain']

    if init_type not in validate_init_types:
        raise GraphiteSendException(
            "Invalid init_type '%s', must be one of: %s" %
            (init_type, ", ".join(validate_init_types)))

    # Use TCP to send data to the plain text receiver on the graphite server.
    if init_type in ['plaintext_tcp', 'plaintext', 'plain']:
        _module_instance = GraphiteClient(*args, **kwargs)

    # Use TCP to send pickled data to the pickle receiver on the graphite
    # server.
    if init_type in ['pickle_tcp', 'pickle']:
        _module_instance = GraphitePickleClient(*args, **kwargs)

    return _module_instance