def colorize(text, color="BLUE", close=True):
    """ Colorizes text for terminal outputs

        @text: #str to colorize
        @color: #str color from :mod:colors
        @close: #bool whether or not to reset the color

        -> #str colorized @text
        ..
            from vital.debug import colorize

            colorize("Hello world", "blue")
            # -> '\x1b[0;34mHello world\x1b[1;m'

            colorize("Hello world", "blue", close=False)
            # -> '\x1b[0;34mHello world'
        ..
    """
    if color:
        color = getattr(colors, color.upper())
        return color + uncolorize(str(text)) + (colors.RESET if close else "")
    return text