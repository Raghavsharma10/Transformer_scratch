def subscribe_to_events(config, subscriber, events, model=None):
    """ Helper function to subscribe to group of events.

    :param config: Pyramid contig instance.
    :param subscriber: Event subscriber function.
    :param events: Sequence of events to subscribe to.
    :param model: Model predicate value.
    """
    kwargs = {}
    if model is not None:
        kwargs['model'] = model

    for evt in events:
        config.add_subscriber(subscriber, evt, **kwargs)