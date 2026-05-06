def _connect_model_signals(model):
    """Connect signals for a single model."""
    dispatch_uid = "%s.post_save" % model._meta.model_name
    logger.debug("Connecting search index model post_save signal: %s", dispatch_uid)
    signals.post_save.connect(_on_model_save, sender=model, dispatch_uid=dispatch_uid)
    dispatch_uid = "%s.post_delete" % model._meta.model_name
    logger.debug("Connecting search index model post_delete signal: %s", dispatch_uid)
    signals.post_delete.connect(
        _on_model_delete, sender=model, dispatch_uid=dispatch_uid
    )