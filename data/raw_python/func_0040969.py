def unsign_filters_and_actions(sign, dotted_model_name):
    """Return the list of filters and actions for dotted_model_name."""
    permissions = signing.loads(sign)
    return permissions.get(dotted_model_name, [])