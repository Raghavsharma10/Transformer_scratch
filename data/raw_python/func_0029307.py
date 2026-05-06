def _trigger_events(view_obj, events_map, additional_kw=None):
    """ Common logic to trigger before/after events.

    :param view_obj: Instance of View that processes the request.
    :param events_map: Map of events from which event class should be
        picked.
    :returns: Instance if triggered event.
    """
    if additional_kw is None:
        additional_kw = {}

    event_kwargs = _get_event_kwargs(view_obj)
    if event_kwargs is None:
        return

    event_kwargs.update(additional_kw)
    event_cls = _get_event_cls(view_obj, events_map)
    event = event_cls(**event_kwargs)
    view_obj.request.registry.notify(event)
    return event