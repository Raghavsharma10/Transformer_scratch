def yesno_to_bool(value, varname):
    """Return True/False from "yes"/"no".

    :param value: template keyword argument value
    :type value: string
    :param varname: name of the variable, for use on exception raising
    :type varname: string
    :raises: :exc:`ImproperlyConfigured`

    Django > 1.5 template boolean/None variables feature.
    """
    if isinstance(value, bool):
        if value:
            value = 'yes'
        else:
            value = 'no'
    elif value is None:
        value = 'no'

    # check value configuration, set boolean value
    if value.lower() in ('yes', 'true'):
        value = True
    elif value.lower() in ('no', 'false'):
        value = False
    else:
        raise ImproperlyConfigured(
            'activeurl: malformed param value for %s' % varname
        )
    return value