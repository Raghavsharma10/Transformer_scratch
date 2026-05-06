def format_arg(value):
    """
    :param value:
        Some value in a dataset.
    :type value:
        varies
    :return:
        unicode representation of that value
    :rtype:
        `unicode`
    """
    translator = repr if isinstance(value, six.string_types) else six.text_type
    return translator(value)