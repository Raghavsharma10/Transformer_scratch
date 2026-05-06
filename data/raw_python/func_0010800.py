def get_label(context, field, obj=None):
    """
    Responsible for figuring out the right label for the passed in field.

    The order of precedence is:
       1) if the view has a field_config and a label specified there, use that label
       2) check for a form in the view, if it contains that field, use it's value
    """
    view = context['view']
    return view.lookup_field_label(context, field, obj)