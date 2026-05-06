def _get_placeholder_arg(arg_name, placeholder):
    """
    Validate and return the Placeholder object that the template variable points to.
    """
    if placeholder is None:
        raise RuntimeWarning(u"placeholder object is None")
    elif isinstance(placeholder, Placeholder):
        return placeholder
    elif isinstance(placeholder, Manager):
        manager = placeholder
        try:
            parent_object = manager.instance  # read RelatedManager code
        except AttributeError:
            parent_object = None

        try:
            placeholder = manager.all()[0]
            if parent_object is not None:
                placeholder.parent = parent_object  # Fill GFK cache
            return placeholder
        except IndexError:
            raise RuntimeWarning(u"No placeholders found for query '{0}.all.0'".format(arg_name))
    else:
        raise ValueError(u"The field '{0}' does not refer to a placeholder object!".format(arg_name))