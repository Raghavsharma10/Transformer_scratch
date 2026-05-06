def _reverse(components, trans):
    """
    Reverse the effect of transformation 'trans' on 'components'
    If the transformation does not affect the components, return the original
    string.
    """

    action, parameter = _get_action(trans)
    comps = list(components)
    string = utils.join(comps)

    if action == _Action.ADD_CHAR and string[-1].lower() == parameter.lower():
        if comps[2]:
            i = 2
        elif comps[1]:
            i = 1
        else:
            i = 0
        comps[i] = comps[i][:-1]
    elif action == _Action.ADD_ACCENT:
        comps = accent.add_accent(comps, Accent.NONE)
    elif action == _Action.ADD_MARK:
        if parameter == Mark.BAR:
            comps[0] = comps[0][:-1] + \
                mark.add_mark_char(comps[0][-1:], Mark.NONE)
        else:
            if mark.is_valid_mark(comps, trans):
                comps[1] = "".join([mark.add_mark_char(c, Mark.NONE)
                                    for c in comps[1]])
    return comps