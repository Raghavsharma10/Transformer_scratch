def set_color(
    fg=Color.normal,
    bg=Color.normal,
    fg_dark=False,
    bg_dark=False,
    underlined=False,
):
    """Set the console color.

    >>> set_color(Color.red, Color.blue)
    >>> set_color('red', 'blue')
    >>> set_color() # returns back to normal
    """
    _set_color(fg, bg, fg_dark, bg_dark, underlined)