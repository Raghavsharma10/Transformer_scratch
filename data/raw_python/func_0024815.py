def _get_meta(model):
    """Return metadata of a model.
    Model could be a real model or evaluated metadata."""
    if isinstance(model, Model):
        w = model.meta
    else:
        w = model  # Already metadata
    return w