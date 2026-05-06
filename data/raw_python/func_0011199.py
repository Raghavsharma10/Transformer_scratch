def dump(context=os.environ):
    """Dump current environment as a dictionary

    Arguments:
        context (dict, optional): Current context, defaults
            to the current environment.

    """

    output = {}
    for key, value in context.iteritems():
        if not key.startswith("BE_"):
            continue
        output[key[3:].lower()] = value

    return output