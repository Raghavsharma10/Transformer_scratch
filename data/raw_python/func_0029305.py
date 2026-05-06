def _get_event_kwargs(view_obj):
    """ Helper function to get event kwargs.

    :param view_obj: Instance of View that processes the request.
    :returns dict: Containing event kwargs or None if events shouldn't
        be fired.
    """
    request = view_obj.request

    view_method = getattr(view_obj, request.action)
    do_trigger = not (
        getattr(view_method, '_silent', False) or
        getattr(view_obj, '_silent', False))

    if do_trigger:
        event_kwargs = {
            'view': view_obj,
            'model': view_obj.Model,
            'fields': FieldData.from_dict(
                view_obj._json_params,
                view_obj.Model)
        }
        ctx = view_obj.context
        if hasattr(ctx, 'pk_field') or isinstance(ctx, DataProxy):
            event_kwargs['instance'] = ctx
        return event_kwargs