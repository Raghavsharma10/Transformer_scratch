def stylize_interactive(text, styles, reset=True):
    """stylize() variant that adds C0 control codes (SOH/STX) for readline
    safety."""
    # problem: readline includes bare ANSI codes in width calculations.
    # solution: wrap nonprinting codes in SOH/STX when necessary.
    # see: https://github.com/dslackw/colored/issues/5
    terminator = _c0wrap(attr("reset")) if reset else ""
    return "{}{}{}".format(_c0wrap(styles), text, terminator)