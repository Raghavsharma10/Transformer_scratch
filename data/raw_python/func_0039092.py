def stylize(txt, bold=False, underline=False):
    """
    Changes style of the text.
    """

    setting = ''
    setting += _SET_BOLD if bold is True else ''
    setting += _SET_UNDERLINE if underline is True else ''
    return setting + str(txt) + _STYLE_RESET