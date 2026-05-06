def get_list_class(context, list):
    """
    Returns the class to use for the passed in list.  We just build something up
    from the object type for the list.
    """
    return "list_%s_%s" % (list.model._meta.app_label, list.model._meta.model_name)