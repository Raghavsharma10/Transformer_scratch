def format_string( string, foreground=None, background=None, reset=True, bold=False,
    faint=False, italic=False, underline=False, blink=False, inverted=False ):
    """Returns a Unicode string formatted with an ANSI escape sequence.

    string
        String to format

    foreground
        Foreground colour to use. Accepted types: None, int (xterm
        palette ID), tuple (RGB, RGBA), Colour

    background
        Background colour to use. Accepted types: None, int (xterm
        palette ID), tuple (RGB, RGBA), Colour

    reset
        Reset the formatting at the end (default: True)

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
    colour_format = format_escape( foreground, background, bold, faint,
                                        italic, underline, blink, inverted )
    reset_format = '' if not reset else ANSI_FORMAT_RESET

    return '{}{}{}'.format( colour_format, string, reset_format )