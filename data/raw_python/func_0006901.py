def format_pixels( top, bottom, reset=True, repeat=1 ):
    """Return the ANSI escape sequence to render two vertically-stacked pixels as a
    single monospace character.

    top
        Top colour to use. Accepted types: None, int (xterm
        palette ID), tuple (RGB, RGBA), Colour

    bottom
        Bottom colour to use. Accepted types: None, int (xterm
        palette ID), tuple (RGB, RGBA), Colour

    reset
        Reset the formatting at the end (default: True)

    repeat
        Number of horizontal pixels to render (default: 1)
    """
    top_src = None
    if isinstance( top, int ):
        top_src = top
    else:
        top_rgba = colour.normalise_rgba( top )
        if top_rgba[3] != 0:
            top_src = top_rgba

    bottom_src = None
    if isinstance( bottom, int ):
        bottom_src = bottom
    else:
        bottom_rgba = colour.normalise_rgba( bottom )
        if bottom_rgba[3] != 0:
            bottom_src = bottom_rgba

    # short circuit for empty pixel
    if (top_src is None) and (bottom_src is None):
        return ' '*repeat 

    string = '▀'*repeat;
    colour_format = []

    if top_src == bottom_src:
        string = '█'*repeat
    elif (top_src is None) and (bottom_src is not None):
        string = '▄'*repeat

    if (top_src is None) and (bottom_src is not None):
        if isinstance( bottom_src, int ):
            colour_format.append( ANSI_FORMAT_FOREGROUND_XTERM_CMD.format( bottom_src ) )
        else:
            colour_format.append( ANSI_FORMAT_FOREGROUND_CMD.format( *bottom_src[:3] ) )
    else:
        if isinstance( top_src, int ):
            colour_format.append( ANSI_FORMAT_FOREGROUND_XTERM_CMD.format( top_src ) )
        else:
            colour_format.append( ANSI_FORMAT_FOREGROUND_CMD.format( *top_src[:3] ) )

    if top_src is not None and bottom_src is not None and top_src != bottom_src:
        if isinstance( top_src, int ):
            colour_format.append( ANSI_FORMAT_BACKGROUND_XTERM_CMD.format( bottom_src ) )
        else:
            colour_format.append( ANSI_FORMAT_BACKGROUND_CMD.format( *bottom_src[:3] ) )

    colour_format = ANSI_FORMAT_BASE.format( ';'.join( colour_format ) )
    reset_format = '' if not reset else ANSI_FORMAT_RESET

    return '{}{}{}'.format( colour_format, string, reset_format )