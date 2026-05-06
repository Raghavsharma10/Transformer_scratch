def colorstart(fgcolor, bgcolor, weight):
    ''' Begin a text style. '''
    if weight:
        weight = bold
    else:
        weight = norm
    if bgcolor:
        out('\x1b[%s;%s;%sm' % (weight, fgcolor, bgcolor))
    else:
        out('\x1b[%s;%sm' % (weight, fgcolor))