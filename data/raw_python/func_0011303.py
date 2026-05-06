def format_kwarg(key, value):
    """
    Return a string of form:  "key=<value>"

    If 'value' is a string, we want it quoted. The goal is to make
    the string a named parameter in a method call.
    """
    translator = repr if isinstance(value, six.string_types) else six.text_type
    arg_value = translator(value)

    return '{0}={1}'.format(key, arg_value)