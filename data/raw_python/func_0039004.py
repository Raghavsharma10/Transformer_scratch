def _extract_modifier(x, i, attrs):
    """Extracts the */+/! modifier in front of the Cite at index 'i' of the
    element list 'x'.  The modifier is stored in 'attrs'.  Returns the updated
    index 'i'."""

    global _cleveref_tex_flag  # pylint: disable=global-statement

    assert x[i]['t'] == 'Cite'
    assert i > 0

    # Check the previous element for a modifier in the last character
    if x[i-1]['t'] == 'Str':
        modifier = x[i-1]['c'][-1]
        if not _cleveref_tex_flag and modifier in ['*', '+']:
            _cleveref_tex_flag = True
        if modifier in ['*', '+', '!']:
            attrs[2].append(['modifier', modifier])
            if len(x[i-1]['c']) > 1:  # Lop the modifier off of the string
                x[i-1]['c'] = x[i-1]['c'][:-1]
            else:  # The element contains only the modifier; delete it
                del x[i-1]
                i -= 1

    return i