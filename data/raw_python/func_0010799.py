def get_class(context, field, obj=None):
    """
    Looks up the class for this field
    """
    view = context['view']
    return view.lookup_field_class(field, obj, "field_" + field)