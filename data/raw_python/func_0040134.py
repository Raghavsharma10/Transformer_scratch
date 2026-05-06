def rainbar(data, maxwidth, incolor=True, hicolor=True,
            cbrackets=('\u2595', '\u258F')):
    ''' Creates a "rainbar" style bar graph. '''
    if not data: return             # Nada to do
    datalen = len(data)
    endpcnt = data[-1][1]
    maxwidth = maxwidth - 2         # because of brackets

    # setup
    csi, csib, _, pal, rst, plen = get_palette(hicolor)

    empty = data[-1][0]
    bucket = float(maxwidth) / plen
    position = 0

    # Print left bracket in correct color:
    if incolor:
        out((csi % pal[0]) + cbrackets[0])  # start bracket
    else:
        out(cbrackets[0])

    for i, part in enumerate(data):
        char, pcnt, fgcolor, bgcolor, bold = part
        if fgcolor and hicolor:
            fgcolor = map8[fgcolor]
        if not bold:
            csib = csi

        lastind = None
        width = int(maxwidth * (pcnt / 100.0))
        offset = position
        position += width

        if incolor:
            for j in range(width):
                # faster?
                colorind = fgcolor or min(int((j+offset)/bucket), (plen-1))
                #~ colorind=fgcolor or get_color_index(j, offset,maxwidth,plen)
                if colorind == lastind:
                    out(char)
                else:
                    color = fgcolor or pal[colorind]
                    out((csib % color) + char)
                lastind = colorind
        else:
            out((char * width))

        if i == (datalen - 1):          # check if last one correct
            if position < maxwidth:
                rest = maxwidth - position
                if incolor:
                    out((csib % pal[-1]) + (empty * rest))
                else:
                    out(char * rest)
            elif position > maxwidth:
                out(chr(8) + ' ' + chr(8))  # backspace

    # Print right bracket in correct color:
    if incolor:
        lastcolor = darkred if (hicolor and endpcnt > 1) else pal[-1]
        out((csi % lastcolor) + cbrackets[1])    # end bracket
        colorend()
    else:
        out(cbrackets[1])