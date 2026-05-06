def is_callable_tag(tag):
    """ Determine whether :tag: is a valid callable string tag.

    String is assumed to be valid callable if it starts with '{{'
    and ends with '}}'.

    :param tag: String name of tag.
    """
    return (isinstance(tag, six.string_types) and
            tag.strip().startswith('{{') and
            tag.strip().endswith('}}'))