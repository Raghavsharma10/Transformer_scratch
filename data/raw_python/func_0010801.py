def get_field_link(context, field, obj=None):
    """
    Determine what the field link should be for the given field, object pair
    """
    view = context['view']
    return view.lookup_field_link(context, field, obj)