def _can_undo(comps, trans_list):
    """
    Return whether a components can be undone with one of the transformation in
    trans_list.
    """
    comps = list(comps)
    accent_list = list(map(accent.get_accent_char, comps[1]))
    mark_list = list(map(mark.get_mark_char, utils.join(comps)))
    action_list = list(map(lambda x: _get_action(x), trans_list))

    def atomic_check(action):
        """
        Check if the `action` created one of the marks, accents, or characters
        in `comps`.
        """
        return (action[0] == _Action.ADD_ACCENT and action[1] in accent_list) \
                or (action[0] == _Action.ADD_MARK and action[1] in mark_list) \
                or (action[0] == _Action.ADD_CHAR and action[1] == \
                    accent.remove_accent_char(comps[1][-1]))  # ơ, ư

    return any(map(atomic_check, action_list))