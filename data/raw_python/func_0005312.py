def get_object_or_none(cls, **kwargs):
    """
    Returns model instance or None if not found.
    :param cls: Class or queryset
    :param kwargs: Filters for get() call
    :return: Object or None
    """
    from django.shortcuts import _get_queryset
    qs = _get_queryset(cls)
    try:
        return qs.get(**kwargs)
    except qs.model.DoesNotExist:
        return None