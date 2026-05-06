def events(config):
    """ Return a generator that yields workflow events.

    For every workflow event that is sent from celery this generator yields an event
    object.

    Args:
        config (Config): Reference to the configuration object from which the
            settings are retrieved.

    Returns:
        generator: A generator that returns workflow events.

    """
    celery_app = create_app(config)

    for event in event_stream(celery_app, filter_by_prefix='task'):
        try:
            yield create_event_model(event)
        except JobEventTypeUnsupported:
            pass