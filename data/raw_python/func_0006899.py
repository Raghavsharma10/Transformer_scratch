def format_escape( foreground=None, background=None, bold=False, faint=False,
    italic=False, underline=False, blink=False, inverted=False ):
    """Returns the ANSI escape sequence to set character formatting.

    foreground
        Foreground colour to use. Accepted types: None, int (xterm
        palette ID), tuple (RGB, RGBA), Colour

    background
        Background colour to use. Accepted types: None, int (xterm
        palette ID), tuple (RGB, RGBA), Colour

    bold
        Enable bold text (default: False)

    faint
        Enable faint text (default: False)

    italic
        Enable italic text (default: False)

    underline
        Enable underlined text (default: False)

    blink
        Enable blinky text (default: False)

    inverted
        Enable inverted text (default: False)
    """
    fg_format = None
    if isinstance( foreground, int ):
        fg_format = ANSI_FORMAT_FOREGROUND_XTERM_CMD.format( foreground )
    else:
        fg_rgba = colour.normalise_rgba( foreground )
        if fg_rgba[3] != 0:
            fg_format = ANSI_FORMAT_FOREGROUND_CMD.format( *fg_rgba[:3] )

    bg_format = None
    if isinstance( background, int ):
        bg_format = ANSI_FORMAT_BACKGROUND_XTERM_CMD.format( background )
    else:
        bg_rgba = colour.normalise_rgba( background )
        if bg_rgba[3] != 0:
            bg_format = ANSI_FORMAT_BACKGROUND_CMD.format( *bg_rgba[:3] )

    colour_format = []
    if fg_format is not None:
        colour_format.append( fg_format )
    if bg_format is not None:
        colour_format.append( bg_format )
    if bold:
        colour_format.append( ANSI_FORMAT_BOLD_CMD )
    if faint:
        colour_format.append( ANSI_FORMAT_FAINT_CMD )
    if italic:
        colour_format.append( ANSI_FORMAT_ITALIC_CMD )
    if underline:
        colour_format.append( ANSI_FORMAT_UNDERLINE_CMD )
    if blink:
        colour_format.append( ANSI_FORMAT_BLINK_CMD )
    if inverted:
        colour_format.append( ANSI_FORMAT_INVERTED_CMD )

    colour_format = ANSI_FORMAT_BASE.format( ';'.join( colour_format ) )
    return colour_format