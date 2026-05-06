def get_sort_field(attr, model):
    """
    Get's the field to sort on for the given
    attr.

    Currently returns attr if it is a field on
    the given model.

    If the models has an attribute matching that name
    and that value has an attribute 'sort_field' than
    that value is used.

    TODO: Provide a way to sort based on a non field
    attribute.
    """

    try:
        if model._meta.get_field(attr):
            return attr
    except FieldDoesNotExist:
        if isinstance(attr, basestring):
            val = getattr(model, attr, None)
            if val and hasattr(val, 'sort_field'):
                return getattr(model, attr).sort_field
        return None