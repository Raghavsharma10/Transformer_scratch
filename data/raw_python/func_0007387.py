def _generate_dom_attrs(attrs, allow_no_value=True):
    """ Yield compiled DOM attribute key-value strings.

    If the value is `True`, then it is treated as no-value. If `None`, then it
    is skipped.
    """
    for attr in iterate_items(attrs):
        if isinstance(attr, basestring):
            attr = (attr, True)
        key, value = attr
        if value is None:
            continue
        if value is True and not allow_no_value:
            value = key  # E.g. <option checked="true" />
        if value is True:
            yield True  # E.g. <option checked />
        else:
            yield '%s="%s"' % (key, value.replace('"', '\\"'))