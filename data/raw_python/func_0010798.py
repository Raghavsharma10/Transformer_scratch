def get_value_from_view(context, field):
    """
    Responsible for deriving the displayed value for the passed in 'field'.

    This first checks for a particular method on the ListView, then looks for a method
    on the object, then finally treats it as an attribute.
    """
    view = context['view']
    obj = None
    if 'object' in context:
        obj = context['object']

    value = view.lookup_field_value(context, obj, field)

    # it's a date
    if type(value) == datetime:
        return format_datetime(value)

    return value