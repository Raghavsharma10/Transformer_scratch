def setup_model_event_subscribers(config, model_cls, schema):
    """ Set up model event subscribers.

    :param config: Pyramid Configurator instance.
    :param model_cls: Model class for which handlers should be connected.
    :param schema: Dict of model JSON schema.
    """
    events_map = get_events_map()
    model_events = schema.get('_event_handlers', {})
    event_kwargs = {'model': model_cls}

    for event_tag, subscribers in model_events.items():
        type_, action = event_tag.split('_')
        event_objects = events_map[type_][action]

        if not isinstance(event_objects, list):
            event_objects = [event_objects]

        for sub_name in subscribers:
            sub_func = resolve_to_callable(sub_name)
            config.subscribe_to_events(
                sub_func, event_objects, **event_kwargs)