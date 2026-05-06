def _get_sampleset(model):
    """Return sampleset of a model or `None` if undefined.
    Model could be a real model or evaluated sampleset."""
    if isinstance(model, Model):
        if hasattr(model, 'sampleset'):
            w = model.sampleset()
        else:
            w = None
    else:
        w = model  # Already a sampleset
    return w