def validate_data_privacy(request, data, wrapper_kw=None):
    """ Validate :data: contains only data allowed by privacy settings.

    :param request: Pyramid Request instance
    :param data: Dict containing request/response data which should be
        validated
    """
    from nefertari import wrappers
    if wrapper_kw is None:
        wrapper_kw = {}

    wrapper = wrappers.apply_privacy(request)
    allowed_fields = wrapper(result=data, **wrapper_kw).keys()
    data = data.copy()
    data.pop('_type', None)
    not_allowed_fields = set(data.keys()) - set(allowed_fields)

    if not_allowed_fields:
        raise wrappers.ValidationError(', '.join(not_allowed_fields))