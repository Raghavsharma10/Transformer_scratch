def _ask(question, default=None, data_type='str', show_hint=False):
    """Interactively ask the user for data"""

    data = default

    if data_type == 'bool':
        data = None
        default_string = "Y" if default else "N"

        while data not in ('Y', 'J', 'N', '1', '0'):
            data = input("%s? [%s]: " % (question, default_string)).upper()

            if data == '':
                return default

        return data in ('Y', 'J', '1')

    elif data_type in ('str', 'unicode'):
        if show_hint:
            msg = "%s? [%s] (%s): " % (question, default, data_type)
        else:
            msg = question

        data = input(msg)

        if len(data) == 0:
            data = default
    elif data_type == 'int':
        if show_hint:
            msg = "%s? [%s] (%s): " % (question, default, data_type)
        else:
            msg = question

        data = input(msg)

        if len(data) == 0:
            data = int(default)
        else:
            data = int(data)

    return data