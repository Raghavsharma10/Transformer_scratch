def _remove_brackets(x, i):
    """Removes curly brackets surrounding the Cite element at index 'i' in
    the element list 'x'.  It is assumed that the modifier has been
    extracted.  Empty strings are deleted from 'x'."""

    assert x[i]['t'] == 'Cite'
    assert i > 0 and i < len(x) - 1

    # Check if the surrounding elements are strings
    if not x[i-1]['t'] == x[i+1]['t'] == 'Str':
        return

    # Trim off curly brackets
    if x[i-1]['c'].endswith('{') and x[i+1]['c'].startswith('}'):
        if len(x[i+1]['c']) > 1:
            x[i+1]['c'] = x[i+1]['c'][1:]
        else:
            del x[i+1]

        if len(x[i-1]['c']) > 1:
            x[i-1]['c'] = x[i-1]['c'][:-1]
        else:
            del x[i-1]