def _get_event_cls(view_obj, events_map):
    """ Helper function to get event class.

    :param view_obj: Instance of View that processes the request.
    :param events_map: Map of events from which event class should be
        picked.
    :returns: Found event class.
    """
    request = view_obj.request
    view_method = getattr(view_obj, request.action)
    event_action = (
        getattr(view_method, '_event_action', None) or
        request.action)
    return events_map[event_action]