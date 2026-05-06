def get_model_string(model):
    """
        This function returns the conventional action designator for a given model.
    """
    name = model if isinstance(model, str) else model.__name__
    return normalize_string(name)