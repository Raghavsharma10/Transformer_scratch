def is_orderable(cls):
    """
    Checks if the provided class is specified as an orderable in
    settings.ORDERABLE_MODELS. If it is return its settings.
    """
    if not getattr(settings, 'ORDERABLE_MODELS', None):
        return False

    labels = resolve_labels(cls)
    if labels['app_model'] in settings.ORDERABLE_MODELS:
        return settings.ORDERABLE_MODELS[labels['app_model']]
    return False