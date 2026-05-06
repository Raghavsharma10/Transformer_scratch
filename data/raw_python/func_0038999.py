def _join_strings(x):
    """Joins adjacent Str elements found in the element list 'x'."""
    for i in range(len(x)-1):  # Process successive pairs of elements
        if x[i]['t'] == 'Str' and x[i+1]['t'] == 'Str':
            x[i]['c'] += x[i+1]['c']
            del x[i+1]  # In-place deletion of element from list
            return None  # Forces processing to repeat
    return True