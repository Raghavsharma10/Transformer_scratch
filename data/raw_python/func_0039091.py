def colorize(txt, fg=None, bg=None):
    """
    Print escape codes to set the terminal color.

    fg and bg are indices into the color palette for the foreground and
    background colors.
    """

    setting = ''
    setting += _SET_FG.format(fg) if fg else ''
    setting += _SET_BG.format(bg) if bg else ''
    return setting + str(txt) + _STYLE_RESET