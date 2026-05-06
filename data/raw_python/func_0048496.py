def produces(*content_types):
    """
    Define content types produced by an endpoint.
    """
    def inner(o):
        if not all(isinstance(content_type, _compat.string_types) for content_type in content_types):
            raise ValueError("In parameter not a valid value.")
        try:
            getattr(o, 'produces').update(content_types)
        except AttributeError:
            setattr(o, 'produces', set(content_types))
        return o
    return inner