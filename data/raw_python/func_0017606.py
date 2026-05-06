def coerce_retention_period(value):
    """
    Coerce a retention period to a Python value.

    :param value: A string containing the text 'always', a number or
                  an expression that can be evaluated to a number.
    :returns: A number or the string 'always'.
    :raises: :exc:`~exceptions.ValueError` when the string can't be coerced.
    """
    # Numbers pass through untouched.
    if not isinstance(value, numbers.Number):
        # Other values are expected to be strings.
        if not isinstance(value, string_types):
            msg = "Expected string, got %s instead!"
            raise ValueError(msg % type(value))
        # Check for the literal string `always'.
        value = value.strip()
        if value.lower() == 'always':
            value = 'always'
        else:
            # Evaluate other strings as expressions.
            value = simple_eval(value)
            if not isinstance(value, numbers.Number):
                msg = "Expected numeric result, got %s instead!"
                raise ValueError(msg % type(value))
    return value