def stylize(text, styles, reset=True):
    """conveniently styles your text as and resets ANSI codes at its end."""
    terminator = attr("reset") if reset else ""
    return "{}{}{}".format("".join(styles), text, terminator)