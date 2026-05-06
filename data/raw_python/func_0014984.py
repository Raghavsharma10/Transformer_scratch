def send(*args, **kwargs):
    """ Make sure that we have an instance of the GraphiteClient.
    Then send the metrics to the graphite server.
    User consumable method.
    """
    if not _module_instance:
        raise GraphiteSendException(
            "Must call graphitesend.init() before sending")

    _module_instance.send(*args, **kwargs)
    return _module_instance