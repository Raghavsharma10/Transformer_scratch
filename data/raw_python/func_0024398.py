def _get_object_menu_models():
    """
    we need to create basic permissions
    for only CRUD enabled models
    """
    from pyoko.conf import settings
    enabled_models = []
    for entry in settings.OBJECT_MENU.values():
        for mdl in entry:
            if 'wf' not in mdl:
                enabled_models.append(mdl['name'])
    return enabled_models