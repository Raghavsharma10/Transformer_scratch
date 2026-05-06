def is_disabled_action(view):
    """
    Checks whether Link action is disabled.
    """
    if not isinstance(view, core_views.ActionsViewSet):
        return False

    action = getattr(view, 'action', None)
    return action in view.disabled_actions if action is not None else False