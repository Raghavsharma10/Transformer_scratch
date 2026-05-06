def flag(text=None, color=None, padding=None, show=True, brackets='⸨⸩'):
    """ Wraps @text in parentheses (), optionally colors and pads and
        prints the text.

        @text: #str text to (flag)
        @color: #str color to :func:colorize the text within
        @padding: #str location of padding from :func:padd
        @show: #bool whether or not to print the text in addition to returning
            it

        -> #str (flagged) text
        ..
            from vital.debug import flag

            flag("Hello world", "blue")
            # -> (Hello world)
            #    '(\x1b[0;34mHello world\x1b[1;m)'

            flag("Hello world", "blue", show=False)
            # -> '(\x1b[0;34mHello world\x1b[1;m)'

            flag("Hello world", color="blue", padding="all")
            # ->
            #    (Hello world)
            #
            #    '\\n(\x1b[0;34mHello world\x1b[1;m)\\n'
        ..
    """
    _flag = None
    if text:
        _flag = padd(
            "{}{}{}".format(
                brackets[0],
                colorize(text, color) if color else text,
                brackets[1]
            ),
            padding
        )
        if not show:
            return _flag
        else:
            print(_flag)
    return _flag or text