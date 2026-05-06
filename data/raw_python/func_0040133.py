def bargraph(data, maxwidth, incolor=True, cbrackets=('\u2595', '\u258F')):
    ''' Creates a monochrome or two-color bar graph. '''
    threshold = 100.0 // (maxwidth * 2)  # if smaller than 1/2 of one char wide
    position = 0
    begpcnt = data[0][1] * 100
    endpcnt = data[-1][1] * 100

    if len(data) < 1: return        # Nada to do
    maxwidth = maxwidth - 2         # because of brackets
    datalen = len(data)

    # Print left bracket in correct color:
    if cbrackets and incolor:       # and not (begpcnt == 0 and endpcnt == 0):
        if begpcnt < threshold: bkcolor = data[-1][2]  # greenbg
        else:                   bkcolor = data[0][2]   # redbg
        cprint(cbrackets[0], data[0][2], bkcolor, None, None)
    else:
        out(cbrackets[0])

    for i, part in enumerate(data):
        # unpack data
        char, pcnt, fgcolor, bgcolor, bold = part
        width = int(round(pcnt/100.0 * maxwidth, 0))
        position = position + width

        # and graph
        if incolor and not (fgcolor is None):
            cprint(char * width, fgcolor, bgcolor, bold, False)
        else:
            out((char * width))

        if i == (datalen - 1):   # correct last one
            if position < maxwidth:
                if incolor:     # char
                    cprint(char * (maxwidth-position), fgcolor, bgcolor,
                           bold, False)
                else:
                    out(char * (maxwidth-position))
            elif position > maxwidth:
                out(chr(8) + ' ' + chr(8))  # backspace

    # Print right bracket in correct color:
    if cbrackets and incolor:
        if endpcnt < threshold: bkcolor = data[0][3]    # redbg
        else:                   bkcolor = data[1][3]    # greenbg
        cprint(cbrackets[1], data[-1][2], bkcolor, None, None)
    else:
        out(cbrackets[1])