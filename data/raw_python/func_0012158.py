def pcolor(text, color, indent=0):
    r"""
    Return a string that once printed is colorized.

    :param text: Text to colorize
    :type  text: string

    :param  color: Color to use, one of :code:`'black'`, :code:`'red'`,
                   :code:`'green'`, :code:`'yellow'`, :code:`'blue'`,
                   :code:`'magenta'`, :code:`'cyan'`, :code:`'white'` or
                   :code:`'none'` (case insensitive)
    :type   color: string

    :param indent: Number of spaces to prefix the output with
    :type  indent: integer

    :rtype: string

    :raises:
     * RuntimeError (Argument \`color\` is not valid)

     * RuntimeError (Argument \`indent\` is not valid)

     * RuntimeError (Argument \`text\` is not valid)

     * ValueError (Unknown color *[color]*)
    """
    esc_dict = {
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "white": 37,
        "none": -1,
    }
    if not isinstance(text, str):
        raise RuntimeError("Argument `text` is not valid")
    if not isinstance(color, str):
        raise RuntimeError("Argument `color` is not valid")
    if not isinstance(indent, int):
        raise RuntimeError("Argument `indent` is not valid")
    color = color.lower()
    if color not in esc_dict:
        raise ValueError("Unknown color {color}".format(color=color))
    if esc_dict[color] != -1:
        return "\033[{color_code}m{indent}{text}\033[0m".format(
            color_code=esc_dict[color], indent=" " * indent, text=text
        )
    return "{indent}{text}".format(indent=" " * indent, text=text)