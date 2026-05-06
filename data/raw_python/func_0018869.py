def get_premises_model():
    """
    Support for custom company premises model
    with developer friendly validation.
    """
    try:
        app_label, model_name = PREMISES_MODEL.split('.')
    except ValueError:
        raise ImproperlyConfigured("OPENINGHOURS_PREMISES_MODEL must be of the"
                                   " form 'app_label.model_name'")
    premises_model = get_model(app_label=app_label, model_name=model_name)
    if premises_model is None:
        raise ImproperlyConfigured("OPENINGHOURS_PREMISES_MODEL refers to"
                                   " model '%s' that has not been installed"
                                   % PREMISES_MODEL)
    return premises_model