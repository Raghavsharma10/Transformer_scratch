def stripQuotes(value):

    """Strip single or double quotes off string; remove embedded quote pairs"""

    if value[:1] == '"':
        value = value[1:]
        if value[-1:] == '"':
            value = value[:-1]
        # replace "" with "
        value = re.sub(_re_doubleq2, '"', value)
    elif value[:1] == "'":
        value = value[1:]
        if value[-1:] == "'":
            value = value[:-1]
        # replace '' with '
        value = re.sub(_re_singleq2, "'", value)
    return value