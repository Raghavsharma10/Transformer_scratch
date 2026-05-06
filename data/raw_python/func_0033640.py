def _get_action(trans):
    """
    Return the action inferred from the transformation `trans`.
    and the parameter going with this action
    An _Action.ADD_MARK goes with a Mark
    while an _Action.ADD_ACCENT goes with an Accent
    """
    # TODO: VIQR-like convention
    mark_action = {
        '^': (_Action.ADD_MARK, Mark.HAT),
        '+': (_Action.ADD_MARK, Mark.BREVE),
        '*': (_Action.ADD_MARK, Mark.HORN),
        '-': (_Action.ADD_MARK, Mark.BAR),
    }

    accent_action = {
        '\\': (_Action.ADD_ACCENT, Accent.GRAVE),
        '/': (_Action.ADD_ACCENT, Accent.ACUTE),
        '?': (_Action.ADD_ACCENT, Accent.HOOK),
        '~': (_Action.ADD_ACCENT, Accent.TIDLE),
        '.': (_Action.ADD_ACCENT, Accent.DOT),
    }

    if trans[0] in ('<', '+'):
        return _Action.ADD_CHAR, trans[1]
    if trans[0] == "_":
        return _Action.UNDO, trans[1:]
    if len(trans) == 2:
        return mark_action[trans[1]]
    else:
        return accent_action[trans[0]]