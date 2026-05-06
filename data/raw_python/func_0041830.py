def truncstr(text, width, align='right'):
    ''' Truncate a string, with trailing ellipsis. '''
    before = after = ''
    if align == 'left':
        truncated = text[-width+1:]
        before = _ellpico
    elif align:
        truncated = text[:width-1]
        after = _ellpico

    return f'{before}{truncated}{after}'