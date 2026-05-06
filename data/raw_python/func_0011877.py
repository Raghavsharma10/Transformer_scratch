def _wrap_with(color_code):
    '''Color wrapper.

    Example:
        >>> blue = _wrap_with('34')
        >>> print(blue('text'))
        \033[34mtext\033[0m
    '''
    def inner(text, bold=False):
        '''Inner color function.'''
        code = color_code
        if bold:
            code = flo("1;{code}")
        return flo('\033[{code}m{text}\033[0m')
    return inner