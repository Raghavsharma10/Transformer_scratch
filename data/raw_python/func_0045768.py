def paint(str,color='r'):
    '''Utility func, for printing colorful logs in console...

    @args:
    --
    str : String to be modified.
    color : color code to which the string will be formed. default is 'r'=RED

    @returns:
    --
    str : final modified string with foreground color as per parameters.

    '''
    if color in switcher:
        str = switcher[color]+str+colorama.Style.RESET_ALL
    return str