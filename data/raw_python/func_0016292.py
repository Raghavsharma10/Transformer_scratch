def get_model_indexes(model):
    """Return list of all indexes in which a model is configured.

    A model may be configured to appear in multiple indexes. This function
    will return the names of the indexes as a list of strings. This is
    useful if you want to know which indexes need updating when a model
    is saved.

    Args:
        model: a Django model class.

    """
    indexes = []
    for index in get_index_names():
        for app_model in get_index_models(index):
            if app_model == model:
                indexes.append(index)
    return indexes