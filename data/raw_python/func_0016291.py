def get_index_models(index):
    """Return list of models configured for a named index.

    Args:
        index: string, the name of the index to look up.

    """
    models = []
    for app_model in get_index_config(index).get("models"):
        app, model = app_model.split(".")
        models.append(apps.get_model(app, model))
    return models