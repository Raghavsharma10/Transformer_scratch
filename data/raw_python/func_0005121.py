def bold(text, close=True):
    """ Bolds text for terminal outputs

        @text: #str to bold
        @close: #bool whether or not to reset the bold flag

        -> #str bolded @text
        ..
            from vital.debug import bold

            bold("Hello world")
            # -> '\x1b[1mHello world\x1b[1;m'

            bold("Hello world", close=False)
            # -> '\x1b[1mHello world'
        ..
    """
    return getattr(colors, "BOLD") + str(text) + \
        (colors.RESET if close else "")