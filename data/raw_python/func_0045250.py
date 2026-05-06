def parse_docstring(docstring):
    '''
    Given a docstring, parse it into a description and epilog part
    '''
    if docstring is None:
        return '', ''

    parts = _DOCSTRING_SPLIT.split(docstring)

    if len(parts) == 1:
        return docstring, ''
    elif len(parts) == 2:
        return parts[0], parts[1]
    else:
        raise TooManySplitsError()