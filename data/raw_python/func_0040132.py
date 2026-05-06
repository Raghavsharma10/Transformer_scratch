def cprint(text, fg=grey, bg=blackbg, w=norm, cr=False, encoding='utf8'):
    ''' Print a string in a specified color style and then return to normal.
        def cprint(text, fg=white, bg=blackbg, w=norm, cr=True):
    '''
    colorstart(fg, bg, w)
    out(text)
    colorend(cr)