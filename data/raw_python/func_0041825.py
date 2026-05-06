def fmtstr(text='', colorstr=None, align='>', trunc=True, width=0, end=' '):
    ''' Formats, justifies, and returns a given string according to
        specifications.
    '''
    colwidth = width or opts.colwidth

    if trunc:
        if len(text) > colwidth:
            text = truncstr(text, colwidth, align=trunc)  # truncate w/ellipsis

    value = f'{text:{align}{colwidth}}'
    if opts.incolor and colorstr:
        return colorstr % value + end
    else:
        return value + end