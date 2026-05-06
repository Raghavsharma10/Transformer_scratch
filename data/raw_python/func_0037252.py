def _check_action(action):
    """check for invalid actions"""
    if isinstance(action, types.StringTypes):
        action = action.lower()

    if action not in ['learn', 'forget', 'report', 'revoke']:
        raise SpamCError('The action option is invalid')
    return action